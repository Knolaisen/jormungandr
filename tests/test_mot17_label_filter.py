"""
Diagnostic: compare the current MOT17 GT filter to the proposed
detection-protocol filter without training.

Current filter (mot17.py:102):           confidence_score != 0
Proposed (MOT17 detection protocol):     confidence_score == 1
                                          AND class == 1 (Pedestrian)
                                          AND visibility >= MIN_VIS

The tests dump human-readable summaries via `print(...)`. Run with `-s`:

    pytest tests/test_mot17_label_filter.py -s -v
"""

from collections import Counter
import os

import pandas as pd
import pytest


GT_COLUMNS = [
    "frame_number",
    "identity_number",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "confidence_score",
    "class",
    "visibility",
]

MOT17_CLASS_NAMES = {
    1: "Pedestrian",
    2: "Person on vehicle",
    3: "Car",
    4: "Bicycle",
    5: "Motorbike",
    6: "Non-mot vehicle",
    7: "Static person",
    8: "Distractor",
    9: "Occluder",
    10: "Occluder on ground",
    11: "Occluder full",
    12: "Reflection",
    13: "Crowd",
}

COCO91_NAMES = {
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
    12: "stop sign",
}

DATA_DIR = os.environ.get("MOT17_DATA_DIR", "./data")
MOT17_TRAIN = os.path.join(DATA_DIR, "MOT17", "train")
MIN_VIS = 0.0


pytestmark = pytest.mark.skipif(
    not os.path.isdir(MOT17_TRAIN),
    reason=f"MOT17 data not present at {MOT17_TRAIN!r} "
    "(override with MOT17_DATA_DIR=...)",
)


def _sequence_dirs() -> list[str]:
    return sorted(
        os.path.join(MOT17_TRAIN, d)
        for d in os.listdir(MOT17_TRAIN)
        if os.path.isdir(os.path.join(MOT17_TRAIN, d))
    )


def _read_gt(seq_dir: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(seq_dir, "gt", "gt.txt"),
        sep=",",
        header=None,
        names=GT_COLUMNS,
    )


def _current_filter(gt: pd.DataFrame) -> pd.DataFrame:
    return gt.query("confidence_score != 0")


def _proposed_filter(gt: pd.DataFrame, min_vis: float = MIN_VIS) -> pd.DataFrame:
    return gt.query(f"confidence_score == 1 and visibility >= {min_vis}")


def test_raw_distribution():
    """Dump class / confidence_score histograms in the raw gt.txt files."""
    class_counts = Counter()
    conf_counts = Counter()
    for seq_dir in _sequence_dirs():
        gt = _read_gt(seq_dir)
        class_counts.update(gt["class"].astype(int).tolist())
        conf_counts.update(gt["confidence_score"].astype(int).tolist())

    print("\n=== Raw gt.txt: class distribution (all sequences) ===")
    for cls, n in sorted(class_counts.items()):
        print(f"  class={cls:>2} ({MOT17_CLASS_NAMES.get(cls, '?'):<22}): {n:>8} rows")
    print("\n=== Raw gt.txt: confidence_score distribution ===")
    for conf, n in sorted(conf_counts.items()):
        print(f"  conf={conf:>2}: {n:>8} rows")


def test_filter_row_counts_per_sequence():
    """Side-by-side row counts per sequence: raw / current filter / proposed."""
    print(
        f"\n{'sequence':<20}{'raw':>10}{'current':>12}"
        f"{'proposed':>12}{'kept of cur':>14}"
    )
    for seq_dir in _sequence_dirs():
        gt = _read_gt(seq_dir)
        raw = len(gt)
        cur = len(_current_filter(gt))
        new = len(_proposed_filter(gt))
        kept = 100.0 * new / cur if cur else 0.0
        print(
            f"{os.path.basename(seq_dir):<20}{raw:>10}{cur:>12}{new:>12}{kept:>13.1f}%"
        )


def test_category_ids_reaching_criterion():
    """What integer category_ids the loader sends to the criterion / COCO eval.

    The current loader (mot17.py:148) uses `category_id = int(cat)` straight
    from the MOT17 `class` column — these collide with COCO-91 ids in the
    classifier and matcher.
    """
    cur_ids = Counter()
    new_ids = Counter()
    for seq_dir in _sequence_dirs():
        gt = _read_gt(seq_dir)
        cur_ids.update(_current_filter(gt)["class"].astype(int).tolist())
        new_ids.update(_proposed_filter(gt)["class"].astype(int).tolist())

    def _dump(name, counter):
        print(f"\n=== category_id reaching criterion ({name}) ===")
        for cid, n in sorted(counter.items()):
            coco_name = COCO91_NAMES.get(cid, "<other>")
            mot_name = MOT17_CLASS_NAMES.get(cid, "?")
            print(
                f"  category_id={cid:>2} "
                f"(MOT17 '{mot_name}'  vs  COCO-91 '{coco_name}'): {n:>8} rows"
            )

    _dump("current filter", cur_ids)
    _dump("proposed filter", new_ids)

    # The proposed filter must collapse the label space to a single id (1).
    assert set(new_ids.keys()) <= {1}, (
        f"proposed filter still emits non-pedestrian ids: {sorted(new_ids)}"
    )


def test_per_frame_object_density():
    """Objects per frame matter for `num_queries` and `eos_coefficient`.

    MOT17 can be very crowded; DETR defaults to 100 queries and
    eos_coefficient=0.1.
    """
    print(
        f"\n{'sequence':<20}{'cur mean':>10}{'cur max':>10}{'new mean':>10}{'new max':>10}"
    )
    for seq_dir in _sequence_dirs():
        gt = _read_gt(seq_dir)
        cur = _current_filter(gt).groupby("frame_number").size()
        new = _proposed_filter(gt).groupby("frame_number").size()
        cur_mean = cur.mean() if len(cur) else 0.0
        cur_max = int(cur.max()) if len(cur) else 0
        new_mean = new.mean() if len(new) else 0.0
        new_max = int(new.max()) if len(new) else 0
        print(
            f"{os.path.basename(seq_dir):<20}"
            f"{cur_mean:>10.1f}{cur_max:>10}{new_mean:>10.1f}{new_max:>10}"
        )


def test_loader_yields_only_pedestrian_with_proposed_filter(monkeypatch):
    """End-to-end: monkey-patch VODDataset.prepare_dataframe to use the
    proposed filter, then read one clip and confirm every label's
    `class_labels` is the COCO-91 `person` id (1) — or empty for frames
    with no visible pedestrian."""
    pytest.importorskip("torch")
    try:
        import jormungandr.datasets.video.mot17 as mot17_mod
    except Exception as e:
        pytest.skip(f"cannot import mot17 module (image processor unavailable?): {e}")

    def patched_prepare(self, seq_dir, gt_path):
        gt = pd.read_csv(gt_path, sep=",", header=None, names=self.columns)
        gt = _proposed_filter(gt).copy()
        if gt.empty:
            self.gt_annotations[seq_dir] = pd.DataFrame(
                columns=["frame_number", "bbox", "category", "visibility"]
            )
            return
        gt.loc[:, "bbox"] = gt[
            ["bbox_left", "bbox_top", "bbox_width", "bbox_height"]
        ].values.tolist()
        gt = gt[["frame_number", "bbox", "class", "visibility"]]
        combined = (
            gt.groupby("frame_number")
            .agg({"bbox": list, "class": list, "visibility": list})
            .reset_index()
            .rename(columns={"class": "category"})
        )
        self.gt_annotations[seq_dir] = combined

    monkeypatch.setattr(mot17_mod.VODDataset, "prepare_dataframe", patched_prepare)

    seq_dirs = _sequence_dirs()[:1]
    ds = mot17_mod.VODDataset(seq_dirs, n_frames=2)
    sample = ds[0]
    batch = mot17_mod._collate_fn_vod([sample])

    cls_ids: set[int] = set()
    boxes_seen = 0
    for lbl in batch["labels"]:
        for c in lbl["class_labels"]:
            cls_ids.add(int(c.item()))
            boxes_seen += 1

    print(
        f"\nclass_labels reaching criterion after patch: "
        f"ids={sorted(cls_ids)}, total_boxes={boxes_seen}"
    )
    assert cls_ids <= {1}, f"expected only COCO person (1), got {sorted(cls_ids)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
