import os
from torch.utils.data import DataLoader
from typing import Callable

from jormungandr.datasets.image.coco import _build_image_datasets, _collate_fn
from jormungandr.datasets.video.mot17 import _build_vod_datasets, _collate_fn_vod
from jormungandr.utils.seed import build_torch_generator, seed_worker


def _build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable,
    generator,
    prefetch_factor: int | None = None,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=len(os.sched_getaffinity(0)),
        pin_memory=True,
    )
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


_DATASET_DEFAULTS = {
    "coco": {
        "dataset_name": "detection-datasets/coco",
        "collate_fn": _collate_fn,
        "train_prefetch_factor": 2,
    },
    "mot17": {
        "dataset_name": "mot17",
        "collate_fn": _collate_fn_vod,
        "batch_size": 1,
        "train_prefetch_factor": None,
    },
}


def create_dataloaders(
    dataset_identifier: str = "coco",
    data_dir: str = "./data/",
    batch_size: int = 2,
    seed: int = 42,
    shuffle: bool = True,
    collate_fn: Callable | None = None,
    subset_size: int | None = None,  # image-only
    val_split: float = 0.2,  # video-only
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for either a HuggingFace image detection
    dataset or a local MOT-style video object-detection dataset.

    `dataset_identifier` picks the pipeline:
      - "coco": `datasets.load_dataset(dataset_name)` + DETR image collator.
      - "mot17": `VODDataset` over `data_dir/<DATASET_NAME>/train/*` + clip collator.

    Per-type defaults (dataset name, collate_fn, batch_size, prefetch_factor)
    live in `_DATASET_DEFAULTS` and can be overridden by the matching kwarg.
    """
    if dataset_identifier not in _DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset_identifier {dataset_identifier!r}; expected 'coco' or 'mot17'."
        )

    defaults = _DATASET_DEFAULTS[dataset_identifier]
    dataset_name = defaults["dataset_name"]
    collate_fn = collate_fn or defaults["collate_fn"]

    if dataset_identifier == "coco":
        train_ds, val_ds = _build_image_datasets(
            dataset_name, data_dir, seed, subset_size
        )
    else:
        train_ds, val_ds = _build_vod_datasets(
            data_dir, dataset_name, batch_size, val_split
        )
        # each batch is a clip of n_frames frames, so we don't want to batch multiple clips together
        batch_size = 1

    train_gen = build_torch_generator(seed)
    val_gen = build_torch_generator(seed + 1)

    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=train_gen,
        prefetch_factor=defaults["train_prefetch_factor"],
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=val_gen,
    )
    return train_loader, val_loader
