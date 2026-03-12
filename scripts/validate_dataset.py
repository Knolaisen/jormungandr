from jormungandr.dataset import create_dataloaders
import torch
from tqdm import tqdm

train_loader, val_loader = create_dataloaders()

train_batches_with_pixel_values_nan = []
train_batches_with_pixel_masks_nan = []
train_batches_with_labels_nan = []

val_batches_with_pixel_values_nan = []
val_batches_with_pixel_masks_nan = []
val_batches_with_labels_nan = []

for i, train_batch in tqdm(enumerate(train_loader)):
    for pixel_values in train_batch["pixel_values"]:
        if torch.isnan(pixel_values).any():
            train_batches_with_pixel_values_nan.append(i)
            print(f"NaN found in training pixel values at batch {i}")

    for pixel_mask in train_batch["pixel_mask"]:
        if torch.isnan(pixel_mask).any():
            train_batches_with_pixel_masks_nan.append(i)
            print(f"NaN found in training pixel masks at batch {i}")

    for labels in train_batch["labels"]:
        for k, v in labels.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    train_batches_with_labels_nan.append(i)
                    print(f"NaN found in training labels at batch {i}, key {k}")

for i, val_batch in tqdm(enumerate(val_loader)):
    for pixel_values in val_batch["pixel_values"]:
        if torch.isnan(pixel_values).any():
            val_batches_with_pixel_values_nan.append(i)
            print(f"NaN found in validation pixel values at batch {i}")

    for pixel_mask in val_batch["pixel_mask"]:
        if torch.isnan(pixel_mask).any():
            val_batches_with_pixel_masks_nan.append(i)
            print(f"NaN found in validation pixel masks at batch {i}")

    for labels in val_batch["labels"]:
        for k, v in labels.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    val_batches_with_labels_nan.append(i)
                    print(f"NaN found in validation labels at batch {i}, key {k}")


print(
    f"Total training batches with NaN pixel values: {len(train_batches_with_pixel_values_nan)}"
)
print(
    f"Total training batches with NaN pixel masks: {len(train_batches_with_pixel_masks_nan)}"
)
print(f"Total training batches with NaN labels: {len(train_batches_with_labels_nan)}")

print(
    f"Total validation batches with NaN pixel values: {len(val_batches_with_pixel_values_nan)}"
)
print(
    f"Total validation batches with NaN pixel masks: {len(val_batches_with_pixel_masks_nan)}"
)
print(f"Total validation batches with NaN labels: {len(val_batches_with_labels_nan)}")

print(f"Training batches with NaN pixel values: {train_batches_with_pixel_values_nan}")
print(f"Training batches with NaN pixel masks: {train_batches_with_pixel_masks_nan}")
print(f"Training batches with NaN labels: {train_batches_with_labels_nan}")
print(f"Validation batches with NaN pixel values: {val_batches_with_pixel_values_nan}")
print(f"Validation batches with NaN pixel masks: {val_batches_with_pixel_masks_nan}")
print(f"Validation batches with NaN labels: {val_batches_with_labels_nan}")
