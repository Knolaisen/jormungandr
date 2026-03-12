from typing import Callable
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

from jormungandr.utils.seed import build_torch_generator, seed_worker

model_name = "facebook/detr-resnet-50"
image_processor = DetrImageProcessor.from_pretrained(model_name)


def _collate_fn(batch):
    # images from dataset (CHW uint8)
    # images = [item["image"] for item in batch]
    images = [_ensure_3ch(item["image"]) for item in batch]

    # build COCO-style annotations per image
    targets = []
    for item in batch:
        # (N, 4) COCO xywh = [x_min, y_min, width, height]
        # Boxes: x, y, width, height
        boxes = item["objects"]["bbox"]
        class_ids = item["objects"]["category"]  # (N,)
        areas = item["objects"].get("area", None)
        iscrowd = item["objects"].get("iscrowd", None)

        annotations = []
        for i in range(boxes.shape[0]):
            ann = {
                "bbox": boxes[i].tolist(),  # COCO expects python lists
                "category_id": int(class_ids[i].item()),
            }
            if areas is not None:
                ann["area"] = float(areas[i].item())
            if iscrowd is not None:
                ann["iscrowd"] = int(iscrowd[i].item())
            else:
                ann["iscrowd"] = 0
            annotations.append(ann)

        targets.append(
            {"image_id": int(item["image_id"].item()), "annotations": annotations}
        )

    encoded = image_processor(images=images, annotations=targets, return_tensors="pt")

    return {
        "pixel_values": encoded["pixel_values"],  # (B, 3, Hmax, Wmax)
        "pixel_mask": encoded["pixel_mask"],  # (B, Hmax, Wmax)
        "labels": encoded["labels"],  # list[dict] length B
    }


def _ensure_3ch(img: torch.Tensor) -> torch.Tensor:
    # img: (C,H,W), uint8
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(img.shape)}")

    c, h, w = img.shape
    if c == 3:
        return img
    if c == 1:
        return img.repeat(3, 1, 1)
    if c == 4:
        return img[:3]  # drop alpha
    raise ValueError(f"Unsupported channel count: {c} (shape {tuple(img.shape)})")


def create_dataloaders(
    dataset_name: str = "detection-datasets/coco",
    cache_dir: str = "../data/",
    batch_size: int = 32,
    seed: int = 42,
    shuffle: bool = True,
    collate_fn: Callable = _collate_fn,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset(dataset_name, cache_dir=cache_dir)

    torch_train_ds = ds["train"].with_format("torch")
    torch_val_ds = ds["val"].with_format("torch")
    train_generator = build_torch_generator(seed)
    val_generator = build_torch_generator(seed + 1)

    train_loader = DataLoader(
        torch_train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=train_generator,
        num_workers=4,  # try 4, 8, maybe 12 depending on CPU
        pin_memory=True,
        persistent_workers=True,  # keeps workers alive between epochs
        prefetch_factor=2,  # tune upward if needed
    )

    val_loader = DataLoader(
        torch_val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=val_generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader
