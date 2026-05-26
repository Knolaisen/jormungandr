"""
Verify the MOT17 per-sequence half-split with `n_frames`-wide temporal buffer
produced by `_build_vod_datasets`.

For each sequence with N frames and clip length n_frames:
    mid           = N // 2
    left_buffer   = n_frames // 2
    right_buffer  = n_frames - left_buffer
    train frames  = sorted_files[0 : mid - left_buffer]
    val frames    = sorted_files[mid + right_buffer : N]

Runs without GPU and without training.
"""

import os

import pytest


DATA_DIR = os.environ.get("MOT17_DATA_DIR", "./data")
MOT17_TRAIN = os.path.join(DATA_DIR, "MOT17", "train")


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


def _frame_files(seq_dir: str) -> list[str]:
    img_dir = os.path.join(seq_dir, "img1")
    return sorted(
        f
        for f in os.listdir(img_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    )


def _expected_train_val_files(
    seq_dir: str, n_frames: int
) -> tuple[list[str], list[str]]:
    files = _frame_files(seq_dir)
    n = len(files)
    mid = n // 2
    left_buffer = n_frames // 2
    right_buffer = n_frames - left_buffer
    train_end = max(0, mid - left_buffer)
    val_start = min(n, mid + right_buffer)
    return files[:train_end], files[val_start:]


@pytest.fixture(scope="module", params=[1, 4, 8])
def datasets(request):
    pytest.importorskip("torch")
    from jormungandr.datasets.video.mot17 import _build_vod_datasets

    n_frames = request.param
    train_ds, val_ds = _build_vod_datasets(
        DATA_DIR, "mot17", n_frames=n_frames
    )
    return n_frames, train_ds, val_ds


def test_train_val_frame_indices_disjoint(datasets):
    """Train and val draw from disjoint frame sets per sequence."""
    _, train_ds, val_ds = datasets

    train_per_seq: dict[str, set[str]] = {}
    val_per_seq: dict[str, set[str]] = {}
    for seq_dir, frames in train_ds.clips:
        train_per_seq.setdefault(seq_dir, set()).update(frames)
    for seq_dir, frames in val_ds.clips:
        val_per_seq.setdefault(seq_dir, set()).update(frames)

    for seq_dir in set(train_per_seq) | set(val_per_seq):
        overlap = train_per_seq.get(seq_dir, set()) & val_per_seq.get(
            seq_dir, set()
        )
        assert not overlap, (
            f"{os.path.basename(seq_dir)} has {len(overlap)} shared frames "
            f"between train and val: {sorted(overlap)[:5]}"
        )


def test_all_sequences_present_in_both_halves(datasets):
    """No sequence should be excluded from either half."""
    _, train_ds, val_ds = datasets
    in_train = {seq for seq, _ in train_ds.clips}
    in_val = {seq for seq, _ in val_ds.clips}
    all_seqs = set(_sequence_dirs())
    assert in_train == all_seqs, (
        "missing from train half: "
        f"{sorted(os.path.basename(s) for s in all_seqs - in_train)}"
    )
    assert in_val == all_seqs, (
        "missing from val half: "
        f"{sorted(os.path.basename(s) for s in all_seqs - in_val)}"
    )


def test_clip_count_matches_expected(datasets):
    """Per-sequence clip counts match the explicit half-split formula."""
    n_frames, train_ds, val_ds = datasets

    actual_train: dict[str, int] = {}
    actual_val: dict[str, int] = {}
    for seq_dir, _ in train_ds.clips:
        actual_train[seq_dir] = actual_train.get(seq_dir, 0) + 1
    for seq_dir, _ in val_ds.clips:
        actual_val[seq_dir] = actual_val.get(seq_dir, 0) + 1

    print(
        f"\nn_frames={n_frames}\n{'sequence':<22}{'train':>10}{'val':>10}{'total':>10}"
    )
    for seq_dir in _sequence_dirs():
        train_files, val_files = _expected_train_val_files(seq_dir, n_frames)
        expected_train = max(0, len(train_files) - n_frames + 1)
        expected_val = max(0, len(val_files) - n_frames + 1)
        a_train = actual_train.get(seq_dir, 0)
        a_val = actual_val.get(seq_dir, 0)
        print(
            f"{os.path.basename(seq_dir):<22}"
            f"{a_train:>10}{a_val:>10}{a_train + a_val:>10}"
        )
        assert a_train == expected_train, (
            f"{os.path.basename(seq_dir)} train clips: expected "
            f"{expected_train}, got {a_train}"
        )
        assert a_val == expected_val, (
            f"{os.path.basename(seq_dir)} val clips: expected "
            f"{expected_val}, got {a_val}"
        )


def test_no_clip_straddles_midpoint(datasets):
    """Every train clip uses only train frames; every val clip only val frames."""
    n_frames, train_ds, val_ds = datasets

    train_sets: dict[str, set[str]] = {}
    val_sets: dict[str, set[str]] = {}
    for seq_dir in _sequence_dirs():
        train_files, val_files = _expected_train_val_files(seq_dir, n_frames)
        train_sets[seq_dir] = set(train_files)
        val_sets[seq_dir] = set(val_files)

    for seq_dir, frames in train_ds.clips:
        bad = [f for f in frames if f not in train_sets[seq_dir]]
        assert not bad, (
            f"train clip in {os.path.basename(seq_dir)} uses non-train "
            f"frames: {bad}"
        )
    for seq_dir, frames in val_ds.clips:
        bad = [f for f in frames if f not in val_sets[seq_dir]]
        assert not bad, (
            f"val clip in {os.path.basename(seq_dir)} uses non-val "
            f"frames: {bad}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
