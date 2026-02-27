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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
def test_mamba_encoder_inference_with_position_embedding():
    batch_size, sequence_length, model_dimension = 2, 64, 16
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")
    embedding = torch.randn(batch_size, model_dimension).to("cuda")

    encoder = MambaEncoder(model_dimension=model_dimension).to("cuda")
    encoder_with_em = MambaEncoder(
        model_dimension=model_dimension, position_embedding=embedding
    ).to("cuda")

    y = encoder(x)
    y_with_em = encoder_with_em(x)

    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert not y.equal(x), "Output should be different from input after encoding"

    assert y_with_em.shape == x.shape, (
        f"Expected output shape {x.shape}, got {y_with_em.shape}"
    )
    assert not y_with_em.equal(x), (
        "Output should be different from input after encoding with position embedding"
    )

    # Check that the outputs are different when using position embedding
    assert not y.equal(y_with_em), (
        "Outputs should be different when using position embedding"
    )


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
