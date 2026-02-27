from jormungandr.backbone import Backbone
import torch


def test_backbone_output_shape():
    backbone = Backbone()
    batch_size, channels, height, width = 2, 3, 224, 224
    pixel_values = torch.randn(batch_size, channels, height, width)

    features, mask = backbone(pixel_values)

    assert features.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {features.shape[0]}"
    )
    assert mask.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {mask.shape[0]}"
    )
