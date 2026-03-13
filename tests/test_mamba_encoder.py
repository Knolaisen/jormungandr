import torch
from jormungandr.encoder import MambaEncoder
from jormungandr.embedder import DetrSinePositionEmbedding
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
    weidth, height = 8, 8
    sequence_length = weidth * height
    batch_size, model_dimension = 2, 16
    embedder_shape = (batch_size, 0, weidth, height)
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")

    embedder = DetrSinePositionEmbedding(num_position_features=model_dimension // 2).to(
        "cuda"
    )
    embedding = embedder(embedder_shape, device="cuda", dtype=x.dtype)
    encoder = MambaEncoder(model_dimension=model_dimension).to("cuda")
    encoder_with_em = MambaEncoder(model_dimension=model_dimension).to("cuda")

    y = encoder(x)
    y_with_em = encoder_with_em(x, embedding)

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
        MambaEncoder(hidden_state_dim=state_expansion_factor)


@pytest.mark.parametrize("num_layers", [-1])
def test_mamba_encoder_with_invalid_num_layers(num_layers):
    with pytest.raises(ValueError):
        MambaEncoder(num_layers=num_layers)
