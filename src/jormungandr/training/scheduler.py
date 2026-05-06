"""
Learning rate scheduler factory for PyTorch optimizers.

Supports StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
ReduceLROnPlateau, and OneCycleLR — all configured through SchedulerConfig.
T_max and total_steps are inferred from epochs and steps_per_epoch when not
explicitly provided in the config.

Functions:
    build_scheduler -- instantiate and return the configured LR scheduler, or None.
"""

from torch import optim
from torch.optim import lr_scheduler

from jormungandr.config.configuration import SchedulerConfig

SCHEDULER_REGISTRY: dict[str, type] = {
    "StepLR": lr_scheduler.StepLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "OneCycleLR": lr_scheduler.OneCycleLR,
}


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: SchedulerConfig | None,
    epochs: int,
    steps_per_epoch: int,
) -> lr_scheduler.LRScheduler | None:
    if scheduler_config is None:
        return None

    name = scheduler_config.name
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{name}'. Choose from: {list(SCHEDULER_REGISTRY.keys())}"
        )
    params = dict(scheduler_config.params)

    # Auto-fill total_steps / T_max from epochs and steps_per_epoch
    if name == "OneCycleLR" and not params.get("total_steps"):
        params["total_steps"] = epochs * steps_per_epoch
    if name == "CosineAnnealingLR" and "T_max" not in params:
        params["T_max"] = epochs

    if (
        name in {"CosineAnnealingWarmRestarts", "CosineAnnealingLR"}
        and "eta_min" in params
    ):
        params["eta_min"] = float(params["eta_min"])

    return SCHEDULER_REGISTRY[name](optimizer, **params)
