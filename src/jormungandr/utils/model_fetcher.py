from functools import lru_cache
from transformers import DetrForObjectDetection


@lru_cache(maxsize=None)
def fetch_detr_model(
    model_name: str = "facebook/detr-resnet-50",
) -> DetrForObjectDetection:
    return DetrForObjectDetection.from_pretrained(model_name)
