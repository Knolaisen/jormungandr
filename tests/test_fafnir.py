from jormungandr.backbone import Backbone
from jormungandr.fafnir import Fafnir
import torch
import pytest


@pytest.mark.parametrize(
    "batch_size, channels, height, width, num_queries",
    [
        (1, 3, 224, 224, 1),
        (2, 3, 100, 100, 50),
        (1, 3, 1000, 1000, 100),
    ],
)
def test_fafnir_forward_pass(batch_size, channels, height, width, num_queries):
    pixel_values = torch.randn(batch_size, channels, height, width)
    backbone = Backbone()

    fafnir = Fafnir(backbone=backbone, num_queries=num_queries)

    outputs = fafnir.forward(pixel_values)

    assert outputs.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {outputs.shape[0]}"
    )
    assert outputs.shape[1] == num_queries, (
        f"Expected number of queries {num_queries}, got {outputs.shape[1]}"
    )
