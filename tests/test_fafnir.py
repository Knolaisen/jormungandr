from jormungandr.backbone import Backbone
from jormungandr.mamba_encoder import MambaEncoder
from jormungandr.fafnir import Fafnir
import torch


def test_fafnir_forward_pass():
    batch_size, channels, height, width = 2, 3, 224, 224
    pixel_values = torch.randn(batch_size, channels, height, width)
    
    backbone = Backbone()
    
    fafnir = Fafnir(backbone=backbone)
    
    outputs = fafnir.forward(pixel_values)
    
    assert outputs.shape[0] == batch_size, f"Expected batch size {batch_size}, got {outputs.shape[0]}"