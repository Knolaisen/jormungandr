import torch
from jormungandr.dataset import create_dataloaders, create_vod_dataloader
# Fixture for the dataloader

import pytest


@pytest.fixture(scope="module")
def loaders():
    return create_dataloaders(batch_size=4)


def test_batch_basic(loaders):
    train_loader, _ = loaders
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


def test_padding_mask_consistency(loaders):
    tol_frac_nonzero = 0.02
    train_loader, _ = loaders
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


def test_mot_dataset():
    train_loader, val_loader = create_vod_dataloader(dataset_name="mot20", n_frames=4)

    # Entire batch must have same width and height
    for batch in train_loader:
        images = batch["pixel_values"]
        targets = batch["labels"]

        B, C, H, W = images.shape
        assert C == 3
        assert len(targets) == B

        break  # just test one batch
