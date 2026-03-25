import numpy as np
import torch
import wandb
from transformers.image_transforms import center_to_corners_format

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def log_validation_images(
    pixel_values: torch.Tensor,  # [B, 3, H, W] normalized
    pixel_mask: torch.Tensor,  # [B, H, W]
    labels: list[dict],
    class_logits: torch.Tensor,  # [B, Q, num_classes+1]
    pred_boxes: torch.Tensor,  # [B, Q, 4] normalized cxcywh
    id_to_name: dict[int, str] = COCO_ID_TO_NAME,
    num_images: int = 8,
    score_threshold: float = 0.01,
) -> list[wandb.Image]:
    probs = class_logits.softmax(-1).cpu()
    pred_boxes = pred_boxes.cpu()
    pixel_values = pixel_values.cpu()
    pixel_mask = pixel_mask.cpu()

    # Best foreground class per query (exclude no-object = last index)
    scores, pred_classes = probs[..., :-1].max(-1)  # [B, Q]

    wandb_images = []

    for b in range(min(num_images, len(labels))):
        # Crop padding: find actual image dimensions from pixel_mask
        actual_h = int(pixel_mask[b].any(dim=-1).sum().item())
        actual_w = int(pixel_mask[b].any(dim=-2).sum().item())

        # Denormalize: (3, H, W) float -> (H, W, 3) uint8.
        # Keep the full padded size so all images in the batch are the same dimensions.
        # Box coordinates are absolute pixels within the actual image region (top-left),
        # so they remain correct regardless of padding on the right/bottom.
        img = (
            pixel_values[b] * _IMAGENET_STD[:, None, None]
            + _IMAGENET_MEAN[:, None, None]
        )
        img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # GT boxes: normalized cxcywh -> pixel xyxy scaled to actual (non-padded) dims
        gt_boxes_norm = labels[b]["boxes"].cpu()
        gt_classes = labels[b]["class_labels"].cpu()

        gt_xyxy = center_to_corners_format(gt_boxes_norm)
        gt_xyxy[:, [0, 2]] *= actual_w
        gt_xyxy[:, [1, 3]] *= actual_h

        gt_box_data = [
            {
                "position": {
                    "minX": float(x1),
                    "maxX": float(x2),
                    "minY": float(y1),
                    "maxY": float(y2),
                },
                "class_id": int(gt_classes[i].item()),
                "box_caption": id_to_name.get(
                    int(gt_classes[i].item()), str(int(gt_classes[i].item()))
                ),
                "domain": "pixel",
            }
            for i, (x1, y1, x2, y2) in enumerate(gt_xyxy.tolist())
        ]

        # Predicted boxes: filter by score threshold, normalized cxcywh -> pixel xyxy
        pred_xyxy = center_to_corners_format(pred_boxes[b])
        pred_xyxy[:, [0, 2]] *= actual_w
        pred_xyxy[:, [1, 3]] *= actual_h

        pred_box_data = []
        for q, (x1, y1, x2, y2) in enumerate(pred_xyxy.tolist()):
            s = float(scores[b, q].item())
            if s < score_threshold:
                continue
            cls_id = int(pred_classes[b, q].item())
            cls_name = id_to_name.get(cls_id, str(cls_id))
            pred_box_data.append(
                {
                    "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
                    "class_id": cls_id,
                    "box_caption": f"{cls_name}: {s:.2f}",
                    "scores": {"confidence": round(s, 3)},
                    "domain": "pixel",
                }
            )

        wandb_images.append(
            wandb.Image(
                img,
                boxes={
                    "ground_truth": {
                        "box_data": gt_box_data,
                        "class_labels": id_to_name,
                    },
                    "predictions": {
                        "box_data": pred_box_data,
                        "class_labels": id_to_name,
                    },
                },
            )
        )

    return wandb_images
