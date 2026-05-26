"""
Verify whether the MOT17 `VODDataset` emits only the Pedestrian class
(MOT17 class 1) or surfaces other GT classes (Car, Bicycle, Distractor, ...)
to the downstream criterion / COCO eval.

The raw MOT17 gt.txt files contain 13 classes (1=Pedestrian, 2=Person on
vehicle, 3=Car, 4=Bicycle, 5=Motorbike, ...). The current
`VODDataset.prepare_dataframe` only filters by `confidence_score != 0`, so
every non-zero-confidence class passes through. These tests assert that
only class 1 reaches the model — when they fail, the message lists which
MOT17 class ids leaked.

Three layers of the loader are probed:

  1. `VODDataset.gt_annotations`  — parsed gt.txt dataframes.
  2. `VODDataset.__getitem__()`   — per-frame `category_id` values.
  3. `_collate_fn_vod`            — `class_labels` tensors reaching the
                                    criterion / COCO eval.

Skipped if MOT17 data is not present locally.

Run with `-s` to see the per-class breakdown:

    pytest tests/test_mot17_vod_classes.py -v -s
"""

from collections import Counter
import os

import pytest


DATA_DIR = os.environ.get("MOT17_DATA_DIR", "./data")
MOT17_TRAIN = os.path.join(DATA_DIR, "MOT17", "train")
PEDESTRIAN_ID = 1

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


def _describe(counter: Counter) -> str:
    if not counter:
        return "<empty>"
    return ", ".join(
        f"{cid}={MOT17_CLASS_NAMES.get(cid, '?')}(n={n})"
        for cid, n in sorted(counter.items())
    )


@pytest.fixture(scope="module")
def vod_dataset():
    pytest.importorskip("torch")
    try:
        from jormungandr.datasets.video.mot17 import VODDataset
    except Exception as e:
        pytest.skip(f"cannot import mot17 module (image processor unavailable?): {e}")
    return VODDataset(_sequence_dirs(), n_frames=2)


def test_prepare_dataframe_emits_only_pedestrian(vod_dataset):
    """`VODDataset.gt_annotations` `category` lists should only contain class 1."""
    ids: Counter = Counter()
    for df in vod_dataset.gt_annotations.values():
        for cats in df["category"]:
            ids.update(int(c) for c in cats)

    print(f"\ngt_annotations category distribution: {_describe(ids)}")
    leaked = {cid for cid in ids if cid != PEDESTRIAN_ID}
    assert not leaked, (
        f"non-pedestrian MOT17 classes leak through prepare_dataframe: "
        f"{_describe(Counter({c: ids[c] for c in leaked}))}"
    )


def test_getitem_category_id_only_pedestrian(vod_dataset):
    """`__getitem__` annotations should have `category_id == 1` only."""
    ids: Counter = Counter()
    n_clips = min(len(vod_dataset), 10)
    for i in range(n_clips):
        sample = vod_dataset[i]
        for frame_label in sample["labels"]:
            for ann in frame_label["annotations"]:
                ids[int(ann["category_id"])] += 1

    print(
        f"\ncategory_id distribution over {n_clips} clips: {_describe(ids)}"
    )
    leaked = {cid for cid in ids if cid != PEDESTRIAN_ID}
    assert not leaked, (
        f"non-pedestrian category_ids surface from __getitem__: "
        f"{_describe(Counter({c: ids[c] for c in leaked}))}"
    )


def test_collated_class_labels_only_pedestrian(vod_dataset):
    """`_collate_fn_vod` should produce `class_labels` containing only id 1."""
    from jormungandr.datasets.video.mot17 import _collate_fn_vod

    n_clips = min(len(vod_dataset), 4)
    samples = [vod_dataset[i] for i in range(n_clips)]
    batch = _collate_fn_vod(samples)

    ids: Counter = Counter()
    for lbl in batch["labels"]:
        ids.update(int(c.item()) for c in lbl["class_labels"])

    print(
        f"\nclass_labels reaching criterion over {n_clips} clips: "
        f"{_describe(ids)}"
    )
    leaked = {cid for cid in ids if cid != PEDESTRIAN_ID}
    assert not leaked, (
        f"non-pedestrian class_labels reach the criterion: "
        f"{_describe(Counter({c: ids[c] for c in leaked}))}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
