"""
The trainer script is responsible for training the Jormungandr model. It includes functions for loading data, defining the training loop, and evaluating the model's performance. The trainer will utilize the output head defined in `output_head.py` to compute the loss and update the model's parameters during training. It will also handle saving and loading model checkpoints, as well as logging training metrics for analysis.
It shall be able to train the model on a given dataset, and evaluate its performance on a validation set. The trainer will also include functionality for hyperparameter tuning and early stopping based on validation performance. Additionally, it will support distributed training across multiple GPUs if available.
It shall be able to customize the training process with different optimizers, learning rate schedulers, and loss functions. The trainer will also include functionality for visualizing training progress and results, such as plotting loss curves and displaying sample predictions on the validation set. Overall, the trainer will be a crucial component in the development and optimization of the Jormungandr model for object detection tasks.
It shall be able to train both the Fafnir and Jormungandr models

It shall use:
* Torch
* Torchvision for data loading and augmentation
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

from tqdm import trange
from typing import Callable
import wandb
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from jormungandr.config.configuration import load_config
from jormungandr.dataset import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.training.criterion import build_criterion
from jormungandr.training.optimizer import build_optimizer

CONFIG = load_config("config.yaml")

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def train(
    config=CONFIG,
):

    model: nn.Module = Fafnir(encoder_type=config.fafnir.encoder.type).to("cuda")

    wandb.watch(model, log="all", log_freq=100)
    training_loader, validation_loader = create_dataloaders(
        batch_size=config.trainer.batch_size
    )

    EPOCHS = config.trainer.epochs
    criterion = build_criterion(config.trainer.loss.name)
    optimizer = build_optimizer(config.trainer.optimizer)

    best_val_loss = float("inf")
    for epoch in trange(EPOCHS, desc="Epochs", unit="epoch"):
        model.train(True)
        avg_loss = train_one_epoch(
            model, training_loader, optimizer, criterion, device="cuda"
        )

        running_val_loss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                validation_image, validation_labels = vdata
                voutputs = model(validation_image)
                val_loss = criterion(voutputs, validation_labels)
                running_val_loss += val_loss

        average_validation_loss = running_val_loss / (i + 1)
        print(f"LOSS train {avg_loss:.3f} valid {average_validation_loss:.3f}")
        wandb.log({"train_loss": avg_loss, "val_loss": average_validation_loss})

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
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    device: torch.device | str,
):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"Batch {i + 1} loss: {last_loss:.3f}")
            running_loss = 0.0

    return last_loss
