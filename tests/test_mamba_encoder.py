import torch
from jormungandr.mamba_encoder import MambaEncoder
import pytest


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
def test_mamba_encoder_inference():
    batch_size, sequence_length, model_dimension = 2, 64, 16
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")
    encoder = MambaEncoder(model_dimension=model_dimension).to("cuda")

    y = encoder(x)
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert not y.equal(x), "Output should be different from input after encoding"


@pytest.mark.parametrize("model_dimension", [0, -16])
def test_mamba_encoder_with_invalid_dimension(model_dimension):
    with pytest.raises(ValueError):
        MambaEncoder(model_dimension=model_dimension)


@pytest.mark.parametrize("state_expansion_factor", [0, -16])
def test_mamba_encoder_with_invalid_state_expansion_factor(state_expansion_factor):
    with pytest.raises(ValueError):
        MambaEncoder(state_expansion_factor=state_expansion_factor)


@pytest.mark.parametrize("num_layers", [-1])
def test_mamba_encoder_with_invalid_num_layers(num_layers):
    with pytest.raises(ValueError):
        MambaEncoder(num_layers=num_layers)
