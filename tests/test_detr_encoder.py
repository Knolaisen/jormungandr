import torch
from jormungandr.encoder import DETREncoder
from jormungandr.embedder import DetrSinePositionEmbedding
import pytest


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
@pytest.mark.parametrize(
    "batch_size, sequence_length, model_dimension",
    [
        (2, 64, 256),
        (4, 128, 256),
        (1, 32, 256),
    ],
)
def test_detr_encoder_inference(batch_size, sequence_length, model_dimension):
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")

    encoder = DETREncoder(model_name="facebook/detr-resnet-50").to("cuda")

    y = encoder(x)

    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
    assert not y.equal(x), "Output should be different from input after encoding"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test"
)
def test_detr_encoder_inference_with_position_embedding():
    weidth, height = 8, 8
    sequence_length = weidth * height
    batch_size, model_dimension = 2, 256
    embedder_shape = (batch_size, 0, weidth, height)
    x = torch.randn(batch_size, sequence_length, model_dimension).to("cuda")

    embedder = DetrSinePositionEmbedding(num_position_features=model_dimension // 2).to(
        "cuda"
    )
    embedding = embedder(embedder_shape, device="cuda", dtype=x.dtype)

    encoder = DETREncoder(model_name="facebook/detr-resnet-50").to("cuda")

    y_with_em = encoder(x, position_embedding=embedding)

    assert y_with_em.shape == x.shape, (
        f"Expected output shape {x.shape}, got {y_with_em.shape}"
    )
    assert not y_with_em.equal(x), (
        "Output should be different from input after encoding with position embedding"
    )
