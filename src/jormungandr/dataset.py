from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor

model_name = "facebook/detr-resnet-50"
image_processor = DetrImageProcessor.from_pretrained(model_name)


class DetrCocoDataset(CocoDetection):
    def __init__(self, image_directory: str, annotation_file_path: str):
        super().__init__(image_directory, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        inputs = self.image_processor(
            images, annotations=annotations, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].squeeze()
        target = inputs["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }