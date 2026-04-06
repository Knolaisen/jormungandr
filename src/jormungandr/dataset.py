import os
from typing import Callable

from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import torchvision.transforms.functional as TF

from jormungandr.utils.image_processors import (
    DetrImageProcessorNoPadBBoxUpdate as DetrImageProcessor,
)
from jormungandr.utils.seed import build_torch_generator, seed_worker
from jormungandr.utils.transforms import make_coco_transforms

model_name = "facebook/detr-resnet-50"
image_processor = DetrImageProcessor.from_pretrained(model_name, force_download=True)


# maps 80-class index -> 91-class index
coco80_to_coco91 = {
    0: 1,  # person
    1: 2,  # bicycle
    2: 3,  # car
    3: 4,  # motorcycle
    4: 5,  # airplane
    5: 6,  # bus
    6: 7,  # train
    7: 8,  # truck
    8: 9,  # boat
    9: 10,  # traffic light
    10: 11,  # fire hydrant
    11: 13,  # stop sign
    12: 14,  # parking meter
    13: 15,  # bench
    14: 16,  # bird
    15: 17,  # cat
    16: 18,  # dog
    17: 19,  # horse
    18: 20,  # sheep
    19: 21,  # cow
    20: 22,  # elephant
    21: 23,  # bear
    22: 24,  # zebra
    23: 25,  # giraffe
    24: 27,  # backpack
    25: 28,  # umbrella
    26: 31,  # handbag
    27: 32,  # tie
    28: 33,  # suitcase
    29: 34,  # frisbee
    30: 35,  # skis
    31: 36,  # snowboard
    32: 37,  # sports ball
    33: 38,  # kite
    34: 39,  # baseball bat
    35: 40,  # baseball glove
    36: 41,  # skateboard
    37: 42,  # surfboard
    38: 43,  # tennis racket
    39: 44,  # bottle
    40: 46,  # wine glass
    41: 47,  # cup
    42: 48,  # fork
    43: 49,  # knife
    44: 50,  # spoon
    45: 51,  # bowl
    46: 52,  # banana
    47: 53,  # apple
    48: 54,  # sandwich
    49: 55,  # orange
    50: 56,  # broccoli
    51: 57,  # carrot
    52: 58,  # hot dog
    53: 59,  # pizza
    54: 60,  # donut
    55: 61,  # cake
    56: 62,  # chair
    57: 63,  # couch
    58: 64,  # potted plant
    59: 65,  # bed
    60: 67,  # dining table
    61: 70,  # toilet
    62: 72,  # tv
    63: 73,  # laptop
    64: 74,  # mouse
    65: 75,  # remote
    66: 76,  # keyboard
    67: 77,  # cell phone
    68: 78,  # microwave
    69: 79,  # oven
    70: 80,  # toaster
    71: 81,  # sink
    72: 82,  # refrigerator
    73: 84,  # book
    74: 85,  # clock
    75: 86,  # vase
    76: 87,  # scissors
    77: 88,  # teddy bear
    78: 89,  # hair drier
    79: 90,  # toothbrush
}


def _make_collate_fn(image_set: str) -> Callable:
    transform = make_coco_transforms(image_set)

    def collate_fn(batch):
        images = []
        targets = []

        for item in batch:
            # Convert CHW uint8 tensor -> PIL image for transforms
            pil_img = TF.to_pil_image(_ensure_3ch(item["image"]))

            # Dataset gives xyxy boxes; transforms expect xyxy
            boxes = item["objects"]["bbox"].float()  # (N, 4) xyxy
            class_ids = item["objects"]["category"]  # (N,) 0-indexed 80-class
            areas = item["objects"].get("area", None)
            iscrowd = item["objects"].get("iscrowd", None)

            target = {
                "boxes": boxes,
                "labels": torch.tensor(
                    [coco80_to_coco91[int(c)] for c in class_ids], dtype=torch.long
                ),
            }
            if areas is not None:
                target["area"] = areas.float()
            if iscrowd is not None:
                target["iscrowd"] = iscrowd

            # Apply spatial augmentations (flip, resize, crop)
            pil_img, target = transform(pil_img, target)

            # Convert augmented xyxy boxes -> COCO xywh for image_processor
            aug_boxes = target["boxes"]  # (M, 4) xyxy; M <= N after crop filtering
            aug_class_ids = target["labels"]
            aug_areas = target.get("area", None)
            aug_iscrowd = target.get("iscrowd", None)

            annotations = []
            for i in range(aug_boxes.shape[0]):
                x1, y1, x2, y2 = aug_boxes[i].tolist()
                ann = {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO xywh
                    "category_id": int(aug_class_ids[i].item()),
                }
                if aug_areas is not None:
                    ann["area"] = float(aug_areas[i].item())
                ann["iscrowd"] = int(aug_iscrowd[i].item()) if aug_iscrowd is not None else 0
                annotations.append(ann)

            images.append(pil_img)
            targets.append(
                {"image_id": int(item["image_id"].item()), "annotations": annotations}
            )

        encoded = image_processor(images=images, annotations=targets, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"],  # (B, 3, Hmax, Wmax)
            "pixel_mask": encoded["pixel_mask"],  # (B, Hmax, Wmax)
            "labels": encoded["labels"],  # list[dict] length B
        }

    return collate_fn


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
    subset_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset(dataset_name, cache_dir=cache_dir)

    torch_train_ds = ds["train"].with_format("torch")
    torch_val_ds = ds["val"].with_format("torch")
    train_generator = build_torch_generator(seed)
    val_generator = build_torch_generator(seed + 1)

    if subset_size is not None:
        torch_train_ds = torch_train_ds.shuffle(seed=seed).select(range(subset_size))
        torch_val_ds = torch_val_ds.shuffle(seed=seed + 1).select(range(subset_size))

    train_loader = DataLoader(
        torch_train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_make_collate_fn("train"),
        worker_init_fn=seed_worker,
        generator=train_generator,
        num_workers=len(os.sched_getaffinity(0)),
        pin_memory=True,
        prefetch_factor=2,  # tune upward if needed
    )
    val_loader = DataLoader(
        torch_val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_collate_fn("val"),
        worker_init_fn=seed_worker,
        generator=val_generator,
        num_workers=len(os.sched_getaffinity(0)),
        pin_memory=True,
    )
    return train_loader, val_loader
