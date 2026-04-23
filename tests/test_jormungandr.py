from jormungandr.jormungandr import Jormungandr
from jormungandr.config.configuration import (
    JormungandrConfig,
)

import torch
import pytest


@pytest.mark.parametrize(
    "num_frames, channels, height, width, num_queries",
    [
        (1, 3, 224, 224, 1),
        (2, 3, 1000, 1000, 50),
        (4, 3, 224, 224, 10),
    ],
)
def test_jormungandr_forward_pass(num_frames, channels, height, width, num_queries):
    pixel_values = torch.randn(num_frames, channels, height, width)

    config = JormungandrConfig()
    config.decoder.num_queries = num_queries

    jormungandr = Jormungandr(config=config)
    class_labels, bbox_coordinates, intermediate_outputs = jormungandr.forward(
        pixel_values
    )

    assert class_labels.shape[0] == num_frames, (
        f"Expected num_frames {num_frames}, got {class_labels.shape[0]}"
    )
    assert class_labels.shape[1] == num_queries, (
        f"Expected number of queries {num_queries}, got {class_labels.shape[1]}"
    )

    assert bbox_coordinates.shape[0] == num_frames, (
        f"Expected num_frames {num_frames}, got {bbox_coordinates.shape[0]}"
    )
    assert bbox_coordinates.shape[1] == num_queries, (
        f"Expected number of queries {num_queries}, got {bbox_coordinates.shape[1]}"
    )
    assert bbox_coordinates.shape[2] == 4, (
        f"Expected bbox coordinates to have 4 values (center_x, center_y, width, height), got {bbox_coordinates.shape[2]}"
    )
