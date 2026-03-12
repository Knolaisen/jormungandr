from collections.abc import Callable
from datetime import datetime

from accelerate import Accelerator
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from jormungandr.config.configuration import Config, load_config
from jormungandr.dataset import create_dataloaders
from jormungandr.fafnir import Fafnir
from jormungandr.training.criterion import build_criterion


def train(
    config: Config | None = None,
    accelerator: Accelerator | None = None,
) -> Fafnir | nn.Module:
    config = config or load_config("config.yaml")
    accelerator = accelerator or Accelerator(
        mixed_precision=config.trainer.accelerate.mixed_precision
    )
    model = Fafnir(
        encoder_type=config.fafnir.encoder.type,
        num_encoder_layers=config.fafnir.encoder.num_layers,
        num_classes=config.fafnir.num_classes,
    )
    training_loader, validation_loader = create_dataloaders(
        batch_size=config.trainer.batch_size,
        seed=config.trainer.seed,
        num_workers=config.trainer.num_workers,
        pin_memory=config.trainer.pin_memory,
        persistent_workers=config.trainer.persistent_workers,
        prefetch_factor=config.trainer.prefetch_factor,
    )

    criterion = build_criterion(config.trainer.loss.name)
    optimizer = AdamW(model.parameters(), lr=config.trainer.learning_rate)
    model, optimizer, training_loader, validation_loader = accelerator.prepare(
        model, optimizer, training_loader, validation_loader
    )

    if accelerator.is_main_process and wandb.run is not None:
        wandb.watch(accelerator.unwrap_model(model), log="all", log_freq=100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_loss = float("inf")

    for epoch in trange(
        config.trainer.epochs,
        desc="Epochs",
        unit="epoch",
        disable=not accelerator.is_local_main_process,
    ):
        average_training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
            device=accelerator.device,
            config=config,
            accelerator=accelerator,
        )
        average_validation_loss = run_validation(
            model,
            validation_loader,
            criterion,
            device=accelerator.device,
            config=config,
            accelerator=accelerator,
        )

        accelerator.print(
            f"LOSS train {average_training_loss:.3f} valid {average_validation_loss:.3f}"
        )
        _maybe_log_metrics(
            {
                "train_loss": average_training_loss,
                "val_loss": average_validation_loss,
                "epoch": epoch + 1,
            },
            accelerator=accelerator,
        )

        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            model_path = f"model_{timestamp}_epoch{epoch + 1}_{best_val_loss:.3f}.pt"
            state_dict = accelerator.get_state_dict(model)

            if accelerator.is_main_process:
                accelerator.save(state_dict, model_path)

                if wandb.run is not None:
                    model_artifact = wandb.Artifact(
                        model_path,
                        type="model",
                    )
                    model_artifact.add_file(model_path)
                    wandb.log_artifact(model_artifact)

        accelerator.wait_for_everyone()

    return model


def train_one_epoch(
    model: Fafnir | nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    device: torch.device | str | None = None,
    config: Config | None = None,
    accelerator: Accelerator | None = None,
) -> float:
    config = config or load_config("config.yaml")
    resolved_device = _resolve_device(model, accelerator=accelerator, device=device)

    if accelerator is None:
        model = model.to(resolved_device)

    model.train(True)
    running_loss = torch.zeros((), device=resolved_device)
    seen_examples = torch.zeros((), device=resolved_device)

    progress = tqdm(
        dataloader,
        desc="Batches",
        unit="batch",
        leave=False,
        disable=not _is_local_main_process(accelerator),
    )

    for batch_index, batch in enumerate(progress, start=1):
        pixel_values, pixel_mask, labels = _move_batch_to_device(
            batch, resolved_device, non_blocking=True
        )
        del pixel_mask
        optimizer.zero_grad(set_to_none=True)

        class_labels, bbox_coordinates = model(pixel_values)
        loss, loss_dict, _ = criterion(
            logits=class_labels,
            labels=labels,
            device=resolved_device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )

        _warn_on_invalid_boxes(bbox_coordinates, accelerator=accelerator)

        if accelerator is not None:
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=config.trainer.gradient_clip_norm
                )
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.trainer.gradient_clip_norm)

        optimizer.step()

        batch_size = torch.tensor(pixel_values.shape[0], device=resolved_device).float()
        running_loss = running_loss + loss.detach().float() * batch_size
        seen_examples = seen_examples + batch_size

        if batch_index % config.trainer.log_interval == 0 or batch_index == len(
            dataloader
        ):
            metrics = {
                "train/batch_loss": _reduce_metric(loss, accelerator=accelerator),
                **{
                    f"train/loss/{name}": _reduce_metric(
                        value, accelerator=accelerator
                    )
                    for name, value in loss_dict.items()
                },
            }
            _maybe_log_metrics(metrics, accelerator=accelerator)

    total_loss = _reduce_tensor(running_loss, accelerator=accelerator, reduction="sum")
    total_examples = _reduce_tensor(
        seen_examples, accelerator=accelerator, reduction="sum"
    )
    return (total_loss / total_examples).item()


@torch.no_grad()
def run_validation(
    model: Fafnir | nn.Module,
    validation_loader: DataLoader,
    criterion: nn.Module | Callable,
    device: torch.device | str | None = None,
    config: Config | None = None,
    accelerator: Accelerator | None = None,
) -> float:
    config = config or load_config("config.yaml")
    resolved_device = _resolve_device(model, accelerator=accelerator, device=device)

    if accelerator is None:
        model = model.to(resolved_device)

    model.eval()
    running_val_loss = torch.zeros((), device=resolved_device)
    seen_examples = torch.zeros((), device=resolved_device)

    progress = tqdm(
        validation_loader,
        desc="Validation",
        unit="batch",
        leave=False,
        disable=not _is_local_main_process(accelerator),
    )

    for batch_index, batch in enumerate(progress, start=1):
        pixel_values, pixel_mask, labels = _move_batch_to_device(
            batch, resolved_device, non_blocking=True
        )
        del pixel_mask

        class_labels, bbox_coordinates = model(pixel_values)
        val_loss, loss_dict, _ = criterion(
            logits=class_labels,
            labels=labels,
            device=resolved_device,
            pred_boxes=bbox_coordinates,
            config=config.trainer.loss,
        )

        batch_size = torch.tensor(pixel_values.shape[0], device=resolved_device).float()
        running_val_loss = running_val_loss + val_loss.detach().float() * batch_size
        seen_examples = seen_examples + batch_size

        if batch_index % config.trainer.log_interval == 0 or batch_index == len(
            validation_loader
        ):
            metrics = {
                "val/batch_loss": _reduce_metric(val_loss, accelerator=accelerator),
                **{
                    f"val/loss/{name}": _reduce_metric(
                        value, accelerator=accelerator
                    )
                    for name, value in loss_dict.items()
                },
            }
            _maybe_log_metrics(metrics, accelerator=accelerator)

    total_val_loss = _reduce_tensor(
        running_val_loss, accelerator=accelerator, reduction="sum"
    )
    total_examples = _reduce_tensor(
        seen_examples, accelerator=accelerator, reduction="sum"
    )
    return (total_val_loss / total_examples).item()


def _resolve_device(
    model: nn.Module,
    accelerator: Accelerator | None,
    device: torch.device | str | None,
) -> torch.device:
    if accelerator is not None:
        return accelerator.device
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _move_batch_to_device(
    batch: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]],
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
    pixel_mask = batch["pixel_mask"].to(device, non_blocking=non_blocking)
    labels = [
        {key: value.to(device, non_blocking=non_blocking) for key, value in label.items()}
        for label in batch["labels"]
    ]
    return pixel_values, pixel_mask, labels


def _warn_on_invalid_boxes(
    bbox_coordinates: torch.Tensor,
    accelerator: Accelerator | None,
) -> None:
    if (bbox_coordinates < 0).any():
        _print_once(
            "Warning: Predicted boxes have negative coordinates.",
            accelerator=accelerator,
        )
    if (bbox_coordinates > 1).any():
        _print_once(
            "Warning: Predicted boxes have coordinates greater than 1.",
            accelerator=accelerator,
        )
    if bbox_coordinates.isnan().any():
        _print_once(
            "Warning: Predicted boxes contain NaN values.",
            accelerator=accelerator,
        )
    if bbox_coordinates.isinf().any():
        _print_once(
            "Warning: Predicted boxes contain Inf values.",
            accelerator=accelerator,
        )


def _reduce_metric(
    value: torch.Tensor | float,
    accelerator: Accelerator | None,
) -> float:
    tensor = (
        value.detach().float()
        if isinstance(value, torch.Tensor)
        else torch.tensor(
            value,
            device=accelerator.device if accelerator is not None else None,
        ).float()
    )
    reduced_tensor = _reduce_tensor(tensor, accelerator=accelerator, reduction="mean")
    return reduced_tensor.item()


def _reduce_tensor(
    tensor: torch.Tensor,
    accelerator: Accelerator | None,
    reduction: str,
) -> torch.Tensor:
    tensor = tensor.detach()
    if accelerator is None:
        return tensor
    return accelerator.reduce(tensor, reduction=reduction)


def _maybe_log_metrics(
    metrics: dict[str, float],
    accelerator: Accelerator | None,
) -> None:
    if accelerator is not None and accelerator.trackers:
        accelerator.log(metrics)
        return

    if wandb.run is not None:
        wandb.log(metrics)


def _print_once(message: str, accelerator: Accelerator | None) -> None:
    if accelerator is not None:
        accelerator.print(message)
        return

    print(message)


def _is_local_main_process(accelerator: Accelerator | None) -> bool:
    return accelerator is None or accelerator.is_local_main_process
