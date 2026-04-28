import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable

from jormungandr.datasets.processor import get_image_processor

image_processor = get_image_processor()


def _collate_fn_vod(batch):
    all_images = []
    all_annotations = []
    for item in batch:
        all_images.extend(item["pixel_values"])
        all_annotations.extend(item["labels"])

    encoded = image_processor(
        images=all_images, annotations=all_annotations, return_tensors="pt"
    )

    for label in encoded["labels"]:
        if "area" in label:
            orig_h, orig_w = label["orig_size"].tolist()
            size_h, size_w = label["size"].tolist()
            label["area"] = label["area"] * (orig_h / size_h) * (orig_w / size_w)

    return {
        "pixel_values": encoded["pixel_values"],
        "pixel_mask": encoded["pixel_mask"],
        "labels": encoded["labels"],
    }


class VODDataset(Dataset):
    # sequence_dirs: list of directories, each containing frames of a video sequence
    # n_frames: number of consecutive frames to include in each clip
    # get_item should return a dict with keys "pixel_values" (list of tensors) and "labels" (list of dicts)
    # "pixel_values" should be a list of tensors of shape (Frames, Channels, Height, Width)
    # "labels" should be a list of dicts, one per frame, each containing the COCO-style annotations for that frame
    def __init__(
        self,
        sequence_dirs: list[str],
        n_frames: int = 4,
        # sampling_strategy: str = "consecutive",  # or "random"
        transform: Callable | None = None,
    ):
        self.n_frames = n_frames
        self.transform = transform
        self.sequence_dirs = sequence_dirs
        self.columns = [
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
        self.clips: list[tuple[str, list[str]]] = []
        self.gt_annotations: dict[str, pd.DataFrame] = {}

        """
        Sequence to sequence predictions: for each sequence, we can create multiple clips by sliding a window of size n_frames across the frames. For example, if a sequence has 10 frames and n_frames=4, we can create the following clips:
            - Clip 1: frames 1-4
            - Clip 2: frames 2-5
            - Clip 3: frames 3-6
        They also apply normal image augmentations: random horizontal flip and random resizing
        """
        """
        sequence_dir
        ├── img1/
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        ├── gt/
        │   └── gt.txt
        └── sequence.ini

        """
        for seq_dir in sequence_dirs:
            img_dir = os.path.join(seq_dir, "img1")
            frame_files = sorted(
                [
                    f
                    for f in os.listdir(img_dir)
                    if f.endswith((".jpg", ".png", ".jpeg"))
                ]
            )
            for i in range(len(frame_files) - n_frames + 1):
                clip_frames = frame_files[i : i + n_frames]
                self.clips.append((seq_dir, clip_frames))
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            self.prepare_dataframe(seq_dir, gt_path)

    def prepare_dataframe(self, seq_dir: str, gt_path: str):
        gt = pd.read_csv(gt_path, sep=",", header=None, names=self.columns)
        filtered_gt = gt.query("confidence_score != 0").copy()
        filtered_gt.loc[:, "bbox"] = filtered_gt[
            ["bbox_left", "bbox_top", "bbox_width", "bbox_height"]
        ].values.tolist()
        filtered_gt = filtered_gt[["frame_number", "bbox", "class", "visibility"]]
        combined = (
            filtered_gt.groupby("frame_number")
            .agg({"bbox": list, "class": list, "visibility": list})
            .reset_index()
        )
        combined = combined.rename(columns={"class": "category"})
        self.gt_annotations[seq_dir] = combined

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        seq_dir: str
        clip_frames: list[str]
        seq_dir, clip_frames = self.clips[index]
        pixel_values = []
        labels = []
        for frame_file in clip_frames:
            image, annotation = self.load_image(seq_dir, frame_file)
            if self.transform is not None:
                image, annotation = self.transform(image, annotation)
            pixel_values.append(image)
            labels.append(annotation)
        # pixel_values = torch.stack(pixel_values)  # (Frames, Channels, Height, Width)
        return {"pixel_values": pixel_values, "labels": labels}

    def load_image(self, seq_dir: str, frame_file: str) -> tuple[Image.Image, dict]:
        frame_path = os.path.join(seq_dir, "img1", frame_file)
        img = Image.open(frame_path).convert("RGB")

        frame_number = int(os.path.splitext(frame_file)[0])
        gt = self.gt_annotations[seq_dir]

        rows = gt[gt["frame_number"] == frame_number]
        if rows.empty:
            return img, {"image_id": frame_number, "annotations": []}

        row = rows.iloc[0]
        annotations = [
            {
                "bbox": bbox,  # [x, y, w, h] absolute pixels
                "category_id": int(cat),
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            }
            for bbox, cat in zip(row["bbox"], row["category"])
        ]
        return img, {"image_id": frame_number, "annotations": annotations}


def _build_vod_datasets(
    data_dir: str,
    dataset_name: str,
    n_frames: int,
    val_split: float,
):
    data_path = os.path.join(data_dir, dataset_name.upper(), "train")
    sequence_dirs = sorted(
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    )
    n_val = max(1, int(len(sequence_dirs) * val_split))
    train_dirs = sequence_dirs[:-n_val]
    val_dirs = sequence_dirs[-n_val:]
    return (
        VODDataset(train_dirs, n_frames=n_frames),
        VODDataset(val_dirs, n_frames=n_frames),
    )
