from jormungandr.utils.image_processors import (
    DetrImageProcessorNoPadBBoxUpdate as DetrImageProcessor,
)

image_processor: DetrImageProcessor | None = None


def get_image_processor(
    model_name: str = "facebook/detr-resnet-50",
) -> DetrImageProcessor:
    global image_processor
    if image_processor is None:
        try:
            image_processor = DetrImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading image processor from hub: {e}")
            print("Attempting to load from local cache...")
            image_processor = DetrImageProcessor.from_pretrained(
                model_name, local_files_only=True
            )
    return image_processor
