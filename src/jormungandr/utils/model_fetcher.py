"""
Cached loader for HuggingFace DETR models.

Results are memoised with functools.lru_cache so each unique combination of
(model_name, is_pre_trained, num_encoder_layers, auxiliary_loss) is fetched
from the Hub only once per process. Subsequent calls return the same object,
avoiding redundant downloads and GPU memory allocations.

Functions:
    fetch_detr_model -- return a DetrForObjectDetection instance, cached by args.
"""

from functools import lru_cache
from transformers import DetrForObjectDetection, DetrConfig


@lru_cache()
def fetch_detr_model(
    model_name: str = "facebook/detr-resnet-50",
    is_pre_trained: bool = True,
    num_encoder_layers: int | None = None,
    auxiliary_loss: bool = False,
) -> DetrForObjectDetection:
    config = DetrConfig.from_pretrained(model_name)
    config.auxiliary_loss = auxiliary_loss

    if num_encoder_layers is not None:
        config.encoder_layers = num_encoder_layers

    if not is_pre_trained:
        return DetrForObjectDetection(config)

    if auxiliary_loss:
        return DetrForObjectDetection.from_pretrained(model_name, config=config)

    return DetrForObjectDetection.from_pretrained(model_name, config=config)
