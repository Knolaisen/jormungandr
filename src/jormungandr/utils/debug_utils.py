import os

import torch
from torch import Tensor, nn


def debug_non_finite_enabled() -> bool:
    return os.getenv("JORMUNGANDR_DEBUG_NAN", "").lower() in {"1", "true", "yes"}


def assert_finite_tensor(name: str, tensor: Tensor | None) -> None:
    if tensor is None:
        return

    detached = tensor.detach()
    if not detached.is_floating_point() and not detached.is_complex():
        return

    if torch.isfinite(detached).all():
        return

    raise RuntimeError(
        f"Non-finite tensor detected at {name}. {_tensor_stats(detached)}"
    )


def assert_module_parameters_finite(
    module: nn.Module,
    module_name: str = "model",
) -> None:
    for parameter_name, parameter in module.named_parameters():
        try:
            assert_finite_tensor(f"{module_name}.{parameter_name}", parameter)
        except RuntimeError as error:
            raise RuntimeError(
                f"Non-finite parameter detected in {module_name}.{parameter_name}. {error}"
            ) from error


def assert_module_gradients_finite(
    module: nn.Module,
    module_name: str = "model",
) -> None:
    for parameter_name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue

        try:
            assert_finite_tensor(
                f"{module_name}.{parameter_name}.grad",
                parameter.grad,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Non-finite gradient detected in {module_name}.{parameter_name}. {error}"
            ) from error


def _tensor_stats(tensor: Tensor) -> str:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_values = detached[finite_mask]

    nan_count = torch.isnan(detached).sum().item()
    posinf_count = torch.isposinf(detached).sum().item()
    neginf_count = torch.isneginf(detached).sum().item()

    if finite_values.numel() == 0:
        finite_min = "n/a"
        finite_max = "n/a"
        finite_mean = "n/a"
    else:
        finite_values = finite_values.float()
        finite_min = f"{finite_values.min().item():.6g}"
        finite_max = f"{finite_values.max().item():.6g}"
        finite_mean = f"{finite_values.mean().item():.6g}"

    return (
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"nan={nan_count} +inf={posinf_count} -inf={neginf_count} "
        f"finite_min={finite_min} finite_max={finite_max} finite_mean={finite_mean}"
    )
