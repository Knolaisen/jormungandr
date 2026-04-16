from torch import TensorType
from transformers import DetrImageProcessor
import numpy as np
from transformers.image_utils import AnnotationType, ChannelDimension
from typing import Iterable


class DetrImageProcessorNoPadBBoxUpdate(DetrImageProcessor):
    """DetrImageProcessor that does not rescale bboxes during padding.

    The upstream processor ties ``update_bboxes`` to ``do_convert_annotations``
    (line 1390 of image_processing_detr.py).  This subclass keeps annotation
    conversion (COCO -> centre-format, normalised to [0,1] w.r.t. the
    *original* image) but prevents the padding step from further rescaling the
    boxes to the padded dimensions.
    """

    def pad(
        self,
        images: list[np.ndarray],
        annotations: AnnotationType | list[AnnotationType] | None = None,
        constant_values: float | Iterable[float] = 0,
        return_pixel_mask: bool = True,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = None,
        input_data_format: str | ChannelDimension | None = None,
        update_bboxes: bool = True,
        pad_size: dict[str, int] | None = None,
    ):
        # Force update_bboxes=False so padding never rescales boxes
        return super().pad(
            images,
            annotations,
            constant_values,
            return_pixel_mask,
            return_tensors,
            data_format,
            input_data_format,
            update_bboxes=False,
            pad_size=pad_size,
        )
