import torch
from jormungandr.encoder import DETREncoder 
from jormungandr.embedder import DetrSinePositionEmbedding
import pytest


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
def test_detr_encoder_inference():
    batch_size, sequence_length, model_dimension = 2, 64, 256 
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")
    
    encoder = DETREncoder(model_name="facebook/detr-resnet-50").to("cuda")

    y = encoder(x)

    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert not y.equal(x), "Output should be different from input after encoding"
