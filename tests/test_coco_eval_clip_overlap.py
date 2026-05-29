"""
Regression tests for the overlapping-clip duplication bug in `CocoEvaluator`.

On the MOT17/video path, `batch_size` is the clip length `n_frames` and clips
are built with a stride-1 sliding window, so each physical frame appears in up
to `n_frames` overlapping clips. `CocoEvaluator.update` is called once per clip,
so without dedup the same frame's GT and predictions are accumulated `n_frames`
times under one `image_id`. Combined with COCOeval's `maxDets=100` cap, that
made `coco/AP` collapse as `n_frames` grew (recall ceiling ~ 100 / (n_frames x
objects_per_frame)) — an eval artifact, not a real model regression.

These tests assert the fix: `CocoEvaluator` dedups by `image_id`, keeping the
first occurrence, so the metrics are invariant to how many times a frame is
re-fed via overlapping clips.

Note on absolute values: a single GT box yields AP=0 under pycocotools' 101-point
recall grid, and even 30 perfect boxes on one image cap at ~0.93 (the same
finite-GT interpolation artifact the wrong-class test documents at ~0.98). So we
use a crowded frame and assert *invariance* across re-feeds, not AP==1.
"""

import pytest
import torch


NUM_FG_CLASSES = 91  # DETR foreground; logits last-axis size is NUM_FG_CLASSES + 1 = 92
N_OBJECTS = 30  # crowded frame, so AP is meaningful (non-degenerate) and non-zero


def _peak_logits(class_indices: list[int]) -> torch.Tensor:
    """Construct logits [B=1, Q=len(class_indices), 92] peaked at each class."""
    q = len(class_indices)
    logits = torch.full((1, q, NUM_FG_CLASSES + 1), -10.0)
    for i, c in enumerate(class_indices):
        logits[0, i, c] = 10.0
    return logits


def _grid_boxes(n: int) -> torch.Tensor:
    """`n` distinct, non-overlapping cxcywh boxes laid out on a grid in [0, 1]."""
    side = int(n**0.5) + 1
    boxes = []
    for i in range(n):
        r, c = divmod(i, side)
        boxes.append([(c + 0.5) / side, (r + 0.5) / side, 0.8 / side, 0.8 / side])
    return torch.tensor(boxes)


def _crowded_frame_label(image_id: int, boxes: torch.Tensor) -> list[dict]:
    return [
        {
            "image_id": torch.tensor(image_id),
            "orig_size": torch.tensor([100, 100]),
            "boxes": boxes,
            "class_labels": torch.tensor([1] * boxes.shape[0]),  # all person
        }
    ]


def _feed_perfect(evaluator, image_id: int, boxes: torch.Tensor) -> None:
    """Add one crowded frame whose predictions exactly match its GT."""
    classes = [1] * boxes.shape[0]
    evaluator.update(_peak_logits(classes), boxes.unsqueeze(0), _crowded_frame_label(image_id, boxes))


def test_clip_overlap_does_not_change_metrics(capsys):
    """A crowded frame re-fed K times (as overlapping clips would) must yield the
    same metrics as feeding it once, and store GT/predictions only once."""
    from jormungandr.training.coco_eval import CocoEvaluator

    boxes = _grid_boxes(N_OBJECTS)

    def metrics_for(k: int) -> tuple[dict, CocoEvaluator]:
        ev = CocoEvaluator()
        for _ in range(k):
            _feed_perfect(ev, image_id=42, boxes=boxes)
        return ev.evaluate(), ev

    base, ev_base = metrics_for(1)
    m4, _ = metrics_for(4)
    m16, ev16 = metrics_for(16)

    with capsys.disabled():
        print("\n--- coco/AP for a crowded frame fed 1x / 4x / 16x (overlapping clips) ---")
        for key in ("coco/AP", "coco/AP50", "coco/AR100"):
            print(f"  {key:>12}:  1x={base[key]:.4f}  4x={m4[key]:.4f}  16x={m16[key]:.4f}")

    # The whole point: metrics are invariant to clip overlap / n_frames.
    for key in ("coco/AP", "coco/AP50", "coco/AP75", "coco/AR100"):
        assert m4[key] == pytest.approx(base[key])
        assert m16[key] == pytest.approx(base[key])

    # And the baseline is a meaningful, non-degenerate score (sanity check).
    assert base["coco/AP"] > 0.85
    assert base["coco/AR100"] > 0.9

    # Dedup: only the first occurrence is stored, regardless of re-feeds.
    assert len(ev16.gt_annotations) == len(ev_base.gt_annotations) == N_OBJECTS
    assert len(ev16.predictions) == len(ev_base.predictions) == N_OBJECTS
    assert ev16._seen_image_ids == {42}


def test_first_occurrence_wins(capsys):
    """When the same frame arrives again (a later overlapping clip), the second
    set of predictions is ignored — even if worse. Metrics match the first
    (max-temporal-context) occurrence."""
    from jormungandr.training.coco_eval import CocoEvaluator

    boxes = _grid_boxes(N_OBJECTS)

    ev = CocoEvaluator()
    _feed_perfect(ev, image_id=3, boxes=boxes)  # first clip: perfect predictions
    metrics_first = ev.evaluate()

    # Second clip for the same frame, with badly-shifted predictions.
    bad_boxes = boxes.clone()
    bad_boxes[:, :2] = 0.02  # collapse all predicted centres into a corner
    ev.update(_peak_logits([1] * N_OBJECTS), bad_boxes.unsqueeze(0), _crowded_frame_label(3, boxes))
    metrics_after = ev.evaluate()

    with capsys.disabled():
        print(f"\n--- first-occurrence-wins: AP first={metrics_first['coco/AP']:.4f} "
              f"after-bad-refeed={metrics_after['coco/AP']:.4f} ---")

    assert metrics_after["coco/AP"] == pytest.approx(metrics_first["coco/AP"])
    assert metrics_first["coco/AP"] > 0.85  # the good first prediction is what counts
    assert len(ev.predictions) == N_OBJECTS  # the bad re-feed was dropped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
