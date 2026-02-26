import torch
from jormungandr.mamba_encoder import MambaEncoder

def test_mamba_encoder():
    batch_size, sequence_length, model_dimension = 2, 64, 16
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")
    encoder = MambaEncoder(model_dimension=model_dimension).to("cuda")

    y = encoder(x)
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert not y.equal(x), "Output should be different from input after encoding"