"""
Training and validation loop for Fafnir and Jormungandr.

Reads config.yaml at import time, builds the model, dataloaders, optimizer, and
scheduler, then runs an epoch loop with per-batch gradient updates. The best
checkpoint (by validation AP) is saved to W&B as a model artifact. Separate
learning rates are used for the backbone, encoder(s), decoder, and output head.

Functions:
    train           -- full training loop; returns the trained model.
    validate        -- run a single validation pass and print results.
    train_one_epoch -- one epoch of forward/backward/optimizer steps.
    run_validation  -- validation pass with COCO eval and W&B image logging.
"""

from tqdm import tqdm, trange
from typing import Callable
import wandb
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from datetime import datetime

from jormungandr.config.configuration import (
    Config,
    FafnirConfig,
    JormungandrConfig,
    load_config,
)
from jormungandr.datasets import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.jormungandr import Jormungandr
from jormungandr.training.criterion import build_criterion
from jormungandr.training.scheduler import build_scheduler
from jormungandr.training.coco_eval import CocoEvaluator
from jormungandr.training.visualization import log_validation_images

CONFIG = load_config("config.yaml")
MODELS_PATH = "models/"
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def train(
    config: Config,
):
    device = "cuda"
    if isinstance(config.model, JormungandrConfig):
        model = Jormungandr(config=config.model).to(device)
    elif isinstance(config.model, FafnirConfig):
        model = Fafnir(config=config.model).to(device)

    if config.model.checkpoint_name is not None:
        try:
            api = wandb.Api()
            artifact = api.artifact(config.model.checkpoint_name, type="model")
            artifact_dir = artifact.download()
            checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Successfully loaded checkpoint from {config.model.checkpoint_name}")
            shutil.rmtree(artifact_dir)  # Clean up the downloaded artifact directory
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with randomly initialized weights.")

    training_loader, validation_loader = create_dataloaders(
        dataset_identifier=config.trainer.dataset_name,
        batch_size=config.trainer.batch_size,
        seed=config.trainer.seed,
        subset_size=config.trainer.subset_size,
    )

    criterion = build_criterion(config.trainer.loss.name)
    params = [
        {
            "name": "backbone",
            "params": model.backbone.parameters(),
            "lr": config.trainer.backbone_learning_rate,
        },
        {
            "name": "decoder",
            "params": model.decoder.parameters(),
            "lr": config.trainer.decoder_learning_rate,
        },
        {
            "name": "output_head",
            "params": model.output_head.parameters(),
            "lr": config.trainer.output_head_learning_rate,
        },
    ]
    if isinstance(config.model, FafnirConfig):
        params.append(
            {
                "name": "encoder",
                "params": model.encoder.parameters(),
                "lr": config.trainer.encoder_learning_rate,
            }
        )
    elif isinstance(config.model, JormungandrConfig):
        params.extend(
            [
                {
                    "name": "spatial_encoder",
                    "params": model.spatial_encoder.parameters(),
                    "lr": config.trainer.encoder_learning_rate,
                },
                {
                    "name": "temporal_encoder",
                    "params": model.temporal_encoder.parameters(),
                    "lr": config.trainer.encoder_learning_rate,
                },
            ]
        )
    optimizer = AdamW(params)
    scheduler = build_scheduler(
        optimizer,
        config.trainer.scheduler,
        epochs=config.trainer.epochs,
        steps_per_epoch=len(training_loader),
    )

    best_val_ap = 0.0
    mamba_variant = (
        config.model.encoder.encoder_type
        if isinstance(config.model, FafnirConfig)
        else config.model.spatial_encoder.encoder_type
    )
    artifact_name = f"{mamba_variant}_{config.trainer.loss.name}_{timestamp}"
    for epoch in trange(config.trainer.epochs, desc="Epochs", unit="epoch"):
        _handle_unfreezing(model, epoch, config)
        average_training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device=device,
            config=config,
        )
        average_validation_loss, average_validation_time, val_ap = run_validation(
            model,
            validation_loader,
            criterion,
            device=device,
            config=config,
        )

        # Step the learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(average_validation_loss)
            else:
                scheduler.step()

        current_lrs = {
            f"lr/group_{i}": group["lr"]
            for i, group in enumerate(optimizer.param_groups)
        }
        wandb.log(
            {
                "avg_train_loss": average_training_loss,
                "avg_val_loss": average_validation_loss,
                "epoch": epoch,
                "avg_val_time": average_validation_time,
                **current_lrs,
            }
        )

        if val_ap > best_val_ap:
            os.remove(artifact_name) if os.path.exists(artifact_name) else None
            best_val_ap = val_ap
            model_path = f"{MODELS_PATH}{artifact_name}"
            torch.save(model.state_dict(), model_path)

            model_artifact = wandb.Artifact(
                artifact_name,
                type="model",
                metadata={
                    "epoch": epoch,
                    "val_ap": best_val_ap,
                },
            )
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

    return model


def _handle_unfreezing(model: Fafnir | Jormungandr, epoch: int, config: Config) -> None:
    if config.model.backbone.freeze_backbone:
        if epoch == config.trainer.epoch_to_unfreeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = True
    if config.model.decoder.freeze_decoder:
        if epoch == config.trainer.epoch_to_unfreeze_decoder:
            for param in model.decoder.parameters():
                param.requires_grad = True
    if config.model.output_head.freeze_prediction_head:
        if epoch == config.trainer.epoch_to_unfreeze_output_head:
            for param in model.output_head.parameters():
                param.requires_grad = True


def train_one_epoch(
    model: Fafnir | Jormungandr,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config,
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
        pixel_values = pixel_values.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)
        labels = [
            {k: v.to(device, non_blocking=True) for k, v in label.items()}
            for label in labels
        ]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        class_labels, bbox_coordinates, intermediate = model.forward(
            pixel_values, pixel_mask
        )

        output_class, output_coord = None, None
        if config.trainer.loss.auxiliary_loss:
            output_class, output_coord = model.output_head.forward(intermediate)

        loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
            outputs_class=output_class,
            outputs_coord=output_coord,
        )

        # Backward pass and optimize
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_loss = loss.item()
        running_loss += batch_loss
        wandb.log(
            {
                "train/batch_loss": batch_loss,
            }
        )
    average_loss = running_loss / (i + 1)
    return average_loss


# Disable gradient computation and reduce memory consumption.
@torch.no_grad()
def run_validation(
    model: Fafnir | Jormungandr,
    validation_loader: DataLoader,
    criterion: nn.Module | Callable,
    device: torch.device | str,
    config: Config = CONFIG,
) -> tuple[float, float, float]:
    running_val_loss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    timings = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    evaluator = CocoEvaluator()
    running_loss_dict: dict[str, float] = {}
    viz_batch: dict | None = None

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        starter.record()
        class_labels, bbox_coordinates, intermediate = model(pixel_values, pixel_mask)
        ender.record()

        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

        output_class, output_coord = None, None
        if config.trainer.loss.auxiliary_loss:
            output_class, output_coord = model.output_head.forward(intermediate)

        val_loss, loss_dict, auxiliary_outputs = criterion(
            logits=class_labels,
            labels=labels,
            device=device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
            outputs_class=output_class,
            outputs_coord=output_coord,
        )

        # Aggregate validation loss and metrics
        batch_loss = val_loss.item()
        running_val_loss += batch_loss

        for k, v in loss_dict.items():
            running_loss_dict[k] = running_loss_dict.get(k, 0.0) + float(v)

        evaluator.update(class_labels, bbox_coordinates, labels)

        # Stash the first batch for image logging
        if viz_batch is None:
            n = config.trainer.num_log_images
            viz_batch = {
                "pixel_values": pixel_values[:n].cpu(),
                "pixel_mask": pixel_mask[:n].cpu(),
                "labels": [{k: v.cpu() for k, v in lbl.items()} for lbl in labels[:n]],
                "class_logits": class_labels[:n].cpu(),
                "pred_boxes": bbox_coordinates[:n].cpu(),
            }

    coco_metrics = evaluator.evaluate()
    average_loss_dict = {k: v / (i + 1) for k, v in running_loss_dict.items()}

    wandb_images = log_validation_images(
        **viz_batch,
        num_images=config.trainer.num_log_images,
        score_threshold=config.trainer.viz_score_threshold,
    )
    wandb.log(
        {
            "val/images": wandb_images,
            **{f"val/loss/{k}": v for k, v in average_loss_dict.items()},
            **{f"val/metrics/{k}": v for k, v in coco_metrics.items()},
        }
    )

    val_ap = coco_metrics.get("coco/AP", 0.0)

    average_time = sum(timings) / len(timings)
    average_val_loss = running_val_loss / (i + 1)
    return average_val_loss, average_time, val_ap


def validate(config: Config) -> None:
    device = "cuda"

    if isinstance(config.model, JormungandrConfig):
        model = Jormungandr(config=config.model).to(device)
    elif isinstance(config.model, FafnirConfig):
        model = Fafnir(config=config.model).to(device)

    if config.model.checkpoint_name is not None:
        api = wandb.Api()
        artifact = api.artifact(config.model.checkpoint_name, type="model")
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Successfully loaded checkpoint from {config.model.checkpoint_name}")
        shutil.rmtree(artifact_dir)  # Clean up the downloaded artifact directory

    training_loader, validation_loader = create_dataloaders(
        dataset_identifier=config.trainer.dataset_name,
        batch_size=config.trainer.val_batch_size,
        seed=config.trainer.seed,
    )

    criterion = build_criterion(config.trainer.loss.name)

    average_validation_loss, average_validation_time, val_ap = run_validation(
        model,
        validation_loader,
        criterion,
        device=device,
        config=config,
    )

    print(f"Average validation loss: {average_validation_loss:.4f}")
    print(f"Average validation time per batch: {average_validation_time:.2f} ms")
    print(f"Average validation AP: {val_ap:.4f}")
