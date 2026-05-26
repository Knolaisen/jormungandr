"""
Empirical answer to: does a single prediction in a class that is NOT present
in GT degrade `coco/AP` for the class that IS present?

Setup:
  - 91 foreground classes + 1 no-object channel (DETR default; logits has 92
    channels along the last axis).
  - 100 images, each with exactly one GT pedestrian box (COCO class 1) in the
    centre of the frame.
  - 100 "perfect" class-1 predictions: same box as GT, score ≈ 1.0.
  - 1 extra prediction in class 5 ("airplane"), arbitrary box, on a 101st
    image with no GT. This is the "false prediction on any other class".

Three configurations are compared:
  A. perfect predictions only (baseline)
  B. perfect predictions + 1 wrong-class prediction (default CocoEvaluator)
  C. perfect predictions + 1 wrong-class prediction, with
     `allowed_class_ids={1,2,3,4}` (the MOT17 eval path)

Hypothesis: B ≡ C ≡ A. pycocotools defaults `params.catIds` to
`coco_gt.getCatIds()` (== {1} here), so detections with category_id ∉ {1}
are filtered out internally before AP is computed — the wrong-class
prediction never enters the class-1 AP calculation. The `allowed_class_ids`
filter just drops it earlier.
"""

import pytest
import torch


NUM_FG_CLASSES = 91  # DETR foreground; logits last-axis size is NUM_FG_CLASSES + 1 = 92


def _peak_logits(class_idx: int) -> torch.Tensor:
    """Construct logits [B=1, Q=1, 92] strongly peaked at `class_idx`."""
    logits = torch.full((1, 1, NUM_FG_CLASSES + 1), -10.0)
    logits[0, 0, class_idx] = 10.0
    return logits


def _populate_evaluator(evaluator, include_wrong_class: bool) -> None:
    """Add 100 perfect class-1 predictions and optionally 1 class-5 prediction."""
    gt_box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # cxcywh, normalized
    gt_class = torch.tensor([1])  # COCO person

    pred_boxes_match = gt_box.unsqueeze(0)  # [B=1, Q=1, 4]

    for image_id in range(100):
        labels = [
            {
                "image_id": torch.tensor(image_id),
                "orig_size": torch.tensor([100, 100]),
                "boxes": gt_box,
                "class_labels": gt_class,
            }
        ]
        evaluator.update(_peak_logits(1), pred_boxes_match, labels)

    if include_wrong_class:
        labels_no_gt = [
            {
                "image_id": torch.tensor(999),
                "orig_size": torch.tensor([100, 100]),
                "boxes": torch.zeros(0, 4),
                "class_labels": torch.zeros(0, dtype=torch.long),
            }
        ]
        pred_boxes_wrong = torch.tensor([[[0.3, 0.3, 0.2, 0.2]]])
        evaluator.update(_peak_logits(5), pred_boxes_wrong, labels_no_gt)


def test_one_wrong_class_prediction_does_not_affect_ap(capsys):
    """100 perfect class-1 predictions; adding 1 class-5 prediction must not
    change class-1 AP. Re-confirms that the eval-side filter has no AP
    impact when GT contains only one class."""
    from jormungandr.training.coco_eval import CocoEvaluator

    ev_perfect = CocoEvaluator()
    _populate_evaluator(ev_perfect, include_wrong_class=False)
    metrics_a = ev_perfect.evaluate()

    ev_polluted = CocoEvaluator()
    _populate_evaluator(ev_polluted, include_wrong_class=True)
    metrics_b = ev_polluted.evaluate()

    ev_filtered = CocoEvaluator(allowed_class_ids=frozenset({1, 2, 3, 4}))
    _populate_evaluator(ev_filtered, include_wrong_class=True)
    metrics_c = ev_filtered.evaluate()

    # Disable pycocotools' captured stdout for the report
    with capsys.disabled():
        print("\n--- coco/AP under three configurations ---")
        for k in ("coco/AP", "coco/AP50", "coco/AP75", "coco/AR100"):
            print(
                f"  {k:>12}:  A_perfect={metrics_a[k]:.4f}   "
                f"B_with_wrong_class={metrics_b[k]:.4f}   "
                f"C_allowed_filter={metrics_c[k]:.4f}"
            )
        print(
            f"  predictions stored:  A={len(ev_perfect.predictions)}   "
            f"B={len(ev_polluted.predictions)}   "
            f"C={len(ev_filtered.predictions)}"
        )

    # 100 perfect predictions don't quite hit 1.000 because pycocotools
    # interpolates AP on a fixed 101-point recall grid (the recall=0 step
    # is not covered by 100 GT samples). Expected ~ 100/101 ≈ 0.9901, in
    # practice ~ 0.9802 due to the right-max smoothing. The absolute value
    # isn't what we're testing — the comparison across A/B/C is.
    assert metrics_a["coco/AP"] > 0.95

    # Adding 1 class-5 prediction must not move class-1 AP — this is the
    # core empirical claim.
    assert metrics_b["coco/AP"] == pytest.approx(metrics_a["coco/AP"])
    assert metrics_b["coco/AP50"] == pytest.approx(metrics_a["coco/AP50"])
    assert metrics_b["coco/AR100"] == pytest.approx(metrics_a["coco/AR100"])

    # The allowed_class_ids filter only drops the prediction earlier;
    # numbers must match config B.
    assert metrics_c["coco/AP"] == pytest.approx(metrics_b["coco/AP"])

    # Sanity: the predictions list is shorter under C, identical under A and B.
    assert len(ev_polluted.predictions) == len(ev_perfect.predictions) + 1
    assert len(ev_filtered.predictions) == len(ev_perfect.predictions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
