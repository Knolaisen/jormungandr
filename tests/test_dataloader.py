import torch
from jormungandr.datasets import create_dataloaders
import pytest


@pytest.fixture(scope="module")
def image_loaders():
    return create_dataloaders(batch_size=4)


def test_batch_basic(image_loaders):
    train_loader, _ = image_loaders
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"pixel_values", "pixel_mask", "labels"}

    images = batch["pixel_values"]
    masks = batch["pixel_mask"]
    targets = batch["labels"]

    assert isinstance(images, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert isinstance(targets, list)

    B, C, H, W = images.shape
    assert C == 3
    assert masks.shape == (B, H, W)
    assert len(targets) == B

    assert images.dtype in (torch.float16, torch.float32, torch.bfloat16)
    assert masks.dtype in (torch.bool, torch.uint8, torch.int64, torch.int32)


def test_padding_mask_consistency(image_loaders):
    tol_frac_nonzero = 0.02
    train_loader, _ = image_loaders
    batch = next(iter(train_loader))
    images = batch["pixel_values"]
    masks = batch["pixel_mask"].bool()

    # padded positions: mask == False
    padded = ~masks  # (B,H,W)

    # If nothing padded (rare), pass
    if padded.sum().item() == 0:
        return

    # Check fraction of "significantly nonzero" pixels in padded region
    # images: (B,3,H,W) -> (B,H,W,3)
    img_hw3 = images.permute(0, 2, 3, 1)
    padded_pixels = img_hw3[padded]  # (N,3)

    frac_nonzero = (padded_pixels.abs().mean(dim=1) > 1e-3).float().mean().item()
    assert frac_nonzero < tol_frac_nonzero, (
        f"Too many nonzero padded pixels: {frac_nonzero:.3f}"
    )


@pytest.mark.parametrize("n_frames", [4, 8, 16])
def test_mot_dataset(n_frames):
    train_loader, val_loader = create_dataloaders(
        "mot17", data_dir="./data", batch_size=n_frames
    )

    # Entire batch must have same width and height
    batch = next(iter(train_loader))

    images = batch["pixel_values"]
    targets = batch["labels"]

    B, C, H, W = images.shape
    assert B == n_frames, f"Expected batch size {n_frames}, got {B}"
    assert C == 3, f"Expected 3 RGB channels, got {C}"
    assert len(targets) == B, f"Expected {B} targets, got {len(targets)}"
