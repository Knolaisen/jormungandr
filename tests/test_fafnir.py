from jormungandr.backbone import Backbone
from jormungandr.fafnir import Fafnir
from jormungandr.config.configuration import FafnirConfig, DecoderConfig, EncoderConfig, load_config

import torch
import pytest

config = load_config("test_config.yaml").fafnir

@pytest.mark.parametrize(
    "batch_size, channels, height, width, num_queries, encoder_type",
    [
        (1, 3, 224, 224, 1, "mamba"),
        (2, 3, 100, 100, 50, "mamba"),
        (1, 3, 1000, 1000, 100, "mamba"),
        (1, 3, 224, 224, 1, "detr"),    
        (2, 3, 100, 100, 50, "detr"),
        (1, 3, 1000, 1000, 100, "detr"),
    ],
)
def test_fafnir_forward_pass(
    batch_size, channels, height, width, num_queries, encoder_type
):
    pixel_values = torch.randn(batch_size, channels, height, width)
    backbone = Backbone()

    config.decoder.num_queries = num_queries
    config.encoder.type = encoder_type

    fafnir = Fafnir(
        backbone=backbone, config=config
    )
    class_labels, bbox_coordinates = fafnir.forward(pixel_values)

    assert class_labels.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {class_labels.shape[0]}"
    )
    assert class_labels.shape[1] == num_queries, (
        f"Expected number of queries {num_queries}, got {class_labels.shape[1]}"
    )

    assert bbox_coordinates.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {bbox_coordinates.shape[0]}"
    )
    assert bbox_coordinates.shape[1] == num_queries, (
        f"Expected number of queries {num_queries}, got {bbox_coordinates.shape[1]}"
    )
    assert bbox_coordinates.shape[2] == 4, (
        f"Expected bbox coordinates to have 4 values (center_x, center_y, width, height), got {bbox_coordinates.shape[2]}"
    )
