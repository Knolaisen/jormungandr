from datasets import load_dataset
import torch

from jormungandr.datasets.processor import get_image_processor

image_processor = get_image_processor()

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


def _collate_fn(batch):
    # images from dataset (CHW uint8)
    images = [_ensure_3ch(item["image"]) for item in batch]

    # build COCO-style annotations per image
    targets = []
    for item in batch:
        # dataset bbox format: [x_min, y_min, x_max, y_max] (absolute pixels)
        boxes = item["objects"]["bbox"]
        boxes[:, 2] -= boxes[:, 0]  # width = x_max - x_min
        boxes[:, 3] -= boxes[:, 1]  # height = y_max - y_min

        class_ids = item["objects"]["category"]  # (N,)
        areas = item["objects"].get("area", None)
        iscrowd = item["objects"].get("iscrowd", None)

        annotations = []
        for i in range(boxes.shape[0]):
            ann = {
                "bbox": boxes[i].tolist(),  # COCO expects python lists
                "category_id": coco80_to_coco91[int(class_ids[i].item())],
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

    encoded = image_processor(
        images=images,
        annotations=targets,
        return_tensors="pt",
        # size={"shortest_edge": 1600, "longest_edge": 2666},
    )

    # DetrImageProcessor scales label["area"] by the resize ratio
    # (resize_annotation multiplies by ratio_w * ratio_h). COCO eval size
    # thresholds (small < 32², medium < 96²) are defined in original image
    # pixel space, so we restore each image's areas to original coordinates
    # using label["orig_size"] (original) and label["size"] (after resize).
    for label in encoded["labels"]:
        if "area" in label:
            orig_h, orig_w = label["orig_size"].tolist()
            size_h, size_w = label["size"].tolist()
            label["area"] = label["area"] * (orig_h / size_h) * (orig_w / size_w)

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


def _build_image_datasets(
    dataset_name: str,
    cache_dir: str,
    seed: int,
    subset_size: int | None,
):
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    train_ds = ds["train"].with_format("torch")
    val_ds = ds["val"].with_format("torch")
    if subset_size is not None:
        train_ds = train_ds.shuffle(seed=seed).select(range(subset_size))
        val_ds = val_ds.shuffle(seed=seed + 1).select(range(subset_size))
    return train_ds, val_ds
