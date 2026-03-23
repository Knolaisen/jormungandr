import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers.image_transforms import center_to_corners_format


class CocoEvaluator:
    """Accumulates model predictions and GT annotations across batches, then runs COCOeval."""

    def __init__(self):
        self.gt_images: list[dict] = []
        self.gt_annotations: list[dict] = []
        self.predictions: list[dict] = []
        self._ann_id = 0

    def update(
        self,
        logits: torch.Tensor,      # [B, Q, num_classes+1]
        pred_boxes: torch.Tensor,  # [B, Q, 4] normalized cxcywh
        labels: list[dict],
    ) -> None:
        """Collect predictions and GT annotations for one batch."""
        probs = logits.softmax(-1).detach().cpu()
        pred_boxes = pred_boxes.detach().cpu()

        # Exclude the no-object class (last index); get best foreground class per query
        foreground_scores, pred_classes = probs[..., :-1].max(-1)  # [B, Q]

        for b, label in enumerate(labels):
            image_id = int(label["image_id"].item())
            orig_h, orig_w = label["orig_size"].tolist()

            self.gt_images.append({"id": image_id, "height": int(orig_h), "width": int(orig_w)})

            gt_boxes = label["boxes"].cpu()           # [N, 4] normalized cxcywh
            gt_classes = label["class_labels"].cpu()  # [N] coco91 category IDs
            gt_areas = label.get("area", None)
            gt_iscrowd = label.get("iscrowd", None)

            if len(gt_classes) > 0:
                gt_xywh = _cxcywh_norm_to_xywh_abs(gt_boxes, orig_w, orig_h)
                for i in range(len(gt_classes)):
                    xywh = gt_xywh[i].tolist()
                    self.gt_annotations.append({
                        "id": self._ann_id,
                        "image_id": image_id,
                        "category_id": int(gt_classes[i].item()),
                        "bbox": [round(v, 2) for v in xywh],
                        "iscrowd": int(gt_iscrowd[i].item()) if gt_iscrowd is not None else 0,
                        "area": float(gt_areas[i].item()) if gt_areas is not None else float(xywh[2] * xywh[3]),
                    })
                    self._ann_id += 1

            pb_xywh = _cxcywh_norm_to_xywh_abs(pred_boxes[b], orig_w, orig_h)  # [Q, 4]
            for q in range(pb_xywh.shape[0]):
                self.predictions.append({
                    "image_id": image_id,
                    "category_id": int(pred_classes[b, q].item()),
                    "bbox": [round(v, 2) for v in pb_xywh[q].tolist()],
                    "score": round(float(foreground_scores[b, q].item()), 4),
                })

    def evaluate(self) -> dict[str, float]:
        """Run COCOeval and return a dict of metric name → value."""
        if not self.predictions or not self.gt_annotations:
            return {}

        cat_ids = sorted({ann["category_id"] for ann in self.gt_annotations})

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self.gt_images,
            "annotations": self.gt_annotations,
            "categories": [{"id": cid} for cid in cat_ids],
        }
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self.predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats  # 12-element array
        return {
            "coco/AP":    float(stats[0]),
            "coco/AP50":  float(stats[1]),
            "coco/AP75":  float(stats[2]),
            "coco/APs":   float(stats[3]),
            "coco/APm":   float(stats[4]),
            "coco/APl":   float(stats[5]),
            "coco/AR1":   float(stats[6]),
            "coco/AR10":  float(stats[7]),
            "coco/AR100": float(stats[8]),
            "coco/ARs":   float(stats[9]),
            "coco/ARm":   float(stats[10]),
            "coco/ARl":   float(stats[11]),
        }


def _cxcywh_norm_to_xywh_abs(boxes: torch.Tensor, orig_w: float, orig_h: float) -> torch.Tensor:
    """Convert normalized [cx, cy, w, h] → absolute [x, y, w, h] (COCO bbox format)."""
    if boxes.shape[0] == 0:
        return boxes
    xyxy = center_to_corners_format(boxes)  # [N, 4] normalized x1y1x2y2
    xyxy[:, [0, 2]] *= orig_w
    xyxy[:, [1, 3]] *= orig_h
    xywh = xyxy.clone()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh
