from functools import cache

from jormungandr.utils.image_processors import (
    DetrImageProcessorNoPadBBoxUpdate as DetrImageProcessor,
)


@cache
def get_image_processor(
    model_name: str = "facebook/detr-resnet-50",
) -> DetrImageProcessor:
    try:
        return DetrImageProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading image processor from hub: {e}")
        print("Attempting to load from local cache...")
        return DetrImageProcessor.from_pretrained(model_name, local_files_only=True)
