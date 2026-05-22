"""
Unit tests for the MOT17 -> COCO label mapping and the COCO-eval prediction
filter.

Two boundaries are exercised in isolation, without needing the real MOT17
dataset on disk:

  1. `VODDataset.prepare_dataframe` rewrites the `class` column from MOT17
     ids to COCO-91 ids and drops rows whose MOT17 class has no COCO
     counterpart (Non-mot vehicle, Distractor, Occluder*, Reflection, Crowd).
  2. `CocoEvaluator` with `allowed_class_ids` suppresses any prediction whose
     argmax foreground class is outside the allowed set (treated as no-object).
"""

import os
import tempfile

import pytest
import torch


@pytest.fixture(scope="module")
def mot17_module():
    pytest.importorskip("torch")
    try:
        import jormungandr.datasets.video.mot17 as mod
    except Exception as e:
        pytest.skip(f"cannot import mot17 module (image processor unavailable?): {e}")
    return mod


def _write_synthetic_sequence(root: str, gt_rows: list[str]) -> str:
    """Create a minimal MOT17-style sequence directory with one frame and a
    gt.txt populated by `gt_rows`. Returns the sequence directory path."""
    seq_dir = os.path.join(root, "seq_synthetic")
    os.makedirs(os.path.join(seq_dir, "img1"))
    os.makedirs(os.path.join(seq_dir, "gt"))
    # one tiny placeholder image so VODDataset.__init__ accepts the sequence
    from PIL import Image

    Image.new("RGB", (32, 32)).save(os.path.join(seq_dir, "img1", "000001.jpg"))
    with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as f:
        f.write("\n".join(gt_rows) + "\n")
    return seq_dir


def test_prepare_dataframe_remaps_and_drops(mot17_module):
    """Pedestrian/Person-on-vehicle/Static-person -> 1, Bicycle -> 2, Car -> 3,
    Motorbike -> 4. All other MOT17 classes (with conf=1 for the test) are
    dropped. Rows with conf=0 are also dropped by the existing filter."""
    # gt format: frame, id, x, y, w, h, conf, class, visibility
    gt_rows = [
        # conf=1: kept iff class is in MOT17_TO_COCO
        "1,1,10,10,20,40,1,1,1.0",   # Pedestrian        -> 1
        "1,2,30,30,20,40,1,2,1.0",   # Person on vehicle -> 1
        "1,3,50,50,40,30,1,3,1.0",   # Car               -> 3
        "1,4,80,10,20,20,1,4,1.0",   # Bicycle           -> 2
        "1,5,10,80,30,30,1,5,1.0",   # Motorbike         -> 4
        "1,6, 0, 0, 1, 1,1,6,1.0",   # Non-mot vehicle   -> drop
        "1,7,60,60,20,40,1,7,1.0",   # Static person     -> 1
        "1,8, 0, 0, 1, 1,1,8,1.0",   # Distractor        -> drop
        "1,9, 0, 0, 1, 1,1,9,1.0",   # Occluder          -> drop
        "1,10,0, 0, 1, 1,1,10,1.0",  # Occluder ground   -> drop
        "1,11,0, 0, 1, 1,1,11,1.0",  # Occluder full     -> drop
        "1,12,0, 0, 1, 1,1,12,1.0",  # Reflection        -> drop
        "1,13,0, 0, 1, 1,1,13,1.0",  # Crowd             -> drop
        # conf=0: dropped by the conf filter regardless of class
        "1,14,0, 0, 1, 1,0,1,1.0",   # Pedestrian, conf=0 -> drop
        "1,15,0, 0, 1, 1,0,3,1.0",   # Car, conf=0        -> drop
    ]
    with tempfile.TemporaryDirectory() as tmp:
        seq_dir = _write_synthetic_sequence(tmp, gt_rows)
        ds = mot17_module.VODDataset([seq_dir], n_frames=1)
        df = ds.gt_annotations[seq_dir]
        # one row per frame_number after groupby
        assert len(df) == 1, df
        cats = sorted(int(c) for c in df.iloc[0]["category"])

    # Expected after mapping: {1,1,1, 2, 3, 4} -> sorted [1,1,1,2,3,4]
    assert cats == [1, 1, 1, 2, 3, 4], (
        f"unexpected mapped category list: {cats}"
    )


def test_prepare_dataframe_drops_unmapped_when_conf_nonzero(mot17_module):
    """If a non-mapped MOT17 class somehow has conf!=0, it must still be
    dropped — the new class filter is what guards us from non-MOT17 data
    where conf=0 is not the implicit class filter."""
    gt_rows = [
        "1,1,0,0,1,1,1,9,1.0",   # Occluder, conf=1 -> drop
        "1,2,0,0,1,1,1,13,1.0",  # Crowd, conf=1   -> drop
    ]
    with tempfile.TemporaryDirectory() as tmp:
        seq_dir = _write_synthetic_sequence(tmp, gt_rows)
        ds = mot17_module.VODDataset([seq_dir], n_frames=1)
        df = ds.gt_annotations[seq_dir]
    assert df.empty, f"expected empty df after dropping all rows, got: {df}"


def test_allowed_coco_ids_constant(mot17_module):
    """ALLOWED_COCO_IDS is exactly the image-of MOT17_TO_COCO."""
    assert mot17_module.ALLOWED_COCO_IDS == frozenset({1, 2, 3, 4})


def test_coco_evaluator_filters_disallowed_predictions():
    """Predictions whose argmax foreground class is outside `allowed_class_ids`
    are dropped before being recorded. The COCO path (allowed=None) keeps all
    predictions as before."""
    from jormungandr.training.coco_eval import CocoEvaluator

    num_classes = 91  # DETR default
    # logits[..., :-1] is the foreground; logits[..., -1] is no-object.
    # 4 queries, each peaking at a different foreground class.
    logits = torch.full((1, 4, num_classes + 1), -10.0)
    peak_classes = [1, 5, 3, 50]  # person, airplane, car, hot dog
    for q, c in enumerate(peak_classes):
        logits[0, q, c] = 10.0

    pred_boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]] * 4])
    labels = [
        {
            "image_id": torch.tensor(123),
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.zeros(0, 4),
            "class_labels": torch.zeros(0, dtype=torch.long),
        }
    ]

    ev_default = CocoEvaluator()
    ev_default.update(logits, pred_boxes, labels)
    assert len(ev_default.predictions) == 4, (
        "default CocoEvaluator must keep every query's prediction"
    )

    ev_mot17 = CocoEvaluator(allowed_class_ids=frozenset({1, 2, 3, 4}))
    ev_mot17.update(logits, pred_boxes, labels)
    kept = sorted(p["category_id"] for p in ev_mot17.predictions)
    assert kept == [1, 3], (
        f"MOT17-allowed evaluator must drop classes 5 and 50, kept {kept}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
