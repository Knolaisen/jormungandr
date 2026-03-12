"""
The trainer script is responsible for training the Jormungandr model. It includes functions for loading data, defining the training loop, and evaluating the model's performance. The trainer will utilize the output head defined in `output_head.py` to compute the loss and update the model's parameters during training. It will also handle saving and loading model checkpoints, as well as logging training metrics for analysis.
It shall be able to train the model on a given dataset, and evaluate its performance on a validation set. The trainer will also include functionality for hyperparameter tuning and early stopping based on validation performance. Additionally, it will support distributed training across multiple GPUs if available.
It shall be able to customize the training process with different optimizers, learning rate schedulers, and loss functions. The trainer will also include functionality for visualizing training progress and results, such as plotting loss curves and displaying sample predictions on the validation set. Overall, the trainer will be a crucial component in the development and optimization of the Jormungandr model for object detection tasks.
It shall be able to train both the Fafnir and Jormungandr models

It shall use:
* Torch
* Torchvision for data loading and augmentation
* Freeze backbone
* Learning rate schedular
* Different LR for encoder and decoder
* MultiGPU support Accelerate
* Wandb for logging and visualization
    - Log epoch-wise training and validation loss
    - Log validation metrics
        - mAP, precision, recall, etc.
        - CIoU, GIoU, etc.
    - Log validation images with predicted bounding boxes and labels
    - Log model checkpoints and hyperparameters
        - Learning rate, gradient norms, etc.
    - Watch model gradients and parameters

*
"""

from tqdm import tqdm, trange
from typing import Callable
import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from datetime import datetime

from jormungandr.config.configuration import Config, load_config
from jormungandr.dataset import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.training.criterion import build_criterion

CONFIG = load_config("config.yaml")

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def train(
    config=CONFIG,
):
    device = "cuda"
    model = Fafnir(encoder_type=config.fafnir.encoder.type).to(device)
    wandb.watch(model, log="all", log_freq=100)
    training_loader, validation_loader = create_dataloaders(
        batch_size=config.trainer.batch_size,
        seed=config.trainer.seed,
    )

    EPOCHS = config.trainer.epochs
    criterion = build_criterion(config.trainer.loss.name)
    # optimizer = build_optimizer(config.trainer.optimizer)
    optimizer = AdamW(model.parameters(), lr=config.trainer.learning_rate)

    best_val_loss = float("inf")
    for epoch in trange(EPOCHS, desc="Epochs", unit="epoch"):
        average_training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device=device,
        )
        average_validation_loss = run_validation(
            model,
            validation_loader,
            criterion,
            device=device,
        )

        print(
            f"LOSS train {average_training_loss:.3f} valid {average_validation_loss:.3f}"
        )
        wandb.log(
            {"train_loss": average_training_loss, "val_loss": average_validation_loss}
        )

        # Track best performance, and save the model's state
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            model_path = f"model_{best_val_loss:.3f}_{epoch}"
            torch.save(model.state_dict(), model_path)

            model_artifact = wandb.Artifact(
                model_path,
                type="model",
            )
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

    return model


def train_one_epoch(
    model: Fafnir,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config = CONFIG,
) -> float:
    model.train(True)

    running_loss = 0.0

    for i, data in tqdm(
        enumerate(dataloader), desc="Batches", unit="batch", leave=False
    ):
        pixel_values, pixel_mask, labels = (
            data["pixel_values"],
            data["pixel_mask"],
            data["labels"],
        )
        pixel_values = pixel_values.to(device)
        pixel_mask = pixel_mask.to(device)
        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        class_labels, bbox_coordinates = model.forward(pixel_values)
        loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )

        if (bbox_coordinates < 0).any():
            print(
                "Warning: Predicted boxes have negative coordinates. This may indicate an issue with the predicted boxes."
            )
            print("Predicted boxes:", bbox_coordinates)

        if (bbox_coordinates > 1).any():
            print(
                "Warning: Predicted boxes have coordinates greater than 1. This may indicate an issue with the predicted boxes."
            )
            print("Predicted boxes:", bbox_coordinates)

        if (bbox_coordinates.isnan()).any():
            print(
                "Warning: Predicted boxes contain NaN values. This may indicate an issue with the predicted boxes."
            )
            print("Predicted boxes:", bbox_coordinates)

        if (bbox_coordinates.isinf()).any():
            print(
                "Warning: Predicted boxes contain Inf values. This may indicate an issue with the predicted boxes."
            )
            print("Predicted boxes:", bbox_coordinates)

        # Backward pass and optimize
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss
        wandb.log(
            {
                "train/batch_loss": batch_loss,
                **{f"train/loss/{k}": v for k, v in loss_dict.items()},
                # **{f"batch/aux/{k}": v for k, v in auxiliary_outputs.items()},
            }
        )
    average_loss = running_loss / (i + 1)
    return average_loss


# Disable gradient computation and reduce memory consumption.
@torch.no_grad()
def run_validation(
    model: Fafnir,
    validation_loader: DataLoader,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config = CONFIG,
) -> float:
    running_val_loss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    for i, batch in enumerate(validation_loader):
        pixel_values, pixel_mask, labels = (
            batch["pixel_values"],
            batch["pixel_mask"],
            batch["labels"],
        )
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in label.items()}
            for label in labels
        ]

        class_labels, bbox_coordinates = model.forward(pixel_values)

        val_loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )
        running_val_loss += val_loss

        wandb.log(
            {
                "val/batch_loss": val_loss.item(),
                **{f"val/loss/{k}": v for k, v in loss_dict.items()},
                # **{f"batch/aux/{k}": v for k, v in auxiliary_outputs.items()},
            }
        )
    average_val_loss = running_val_loss / (i + 1)
    return average_val_loss.item()
