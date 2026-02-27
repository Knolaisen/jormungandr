import pytest
import torch

from jormungandr.output_head import MLPPredictionHead


@pytest.mark.parametrize(
    "batch, in_dim, hidden_dim, out_dim, num_layers",
    [
        (1, 8, 16, 4, 1),
        (2, 8, 16, 4, 2),
        (32, 128, 256, 4, 3),
    ],
)
def test_forward_shape_and_dtype(batch, in_dim, hidden_dim, out_dim, num_layers):
    head = MLPPredictionHead(in_dim, hidden_dim, out_dim, num_layers)

    x = torch.randn(batch, in_dim, dtype=torch.float32)
    y = head(x)

    assert y.shape == (batch, out_dim)
    assert y.dtype == x.dtype


@pytest.mark.parametrize("num_layers", [1, 2, 4])
def test_has_expected_number_of_layers(num_layers):
    in_dim, hidden_dim, out_dim = 8, 16, 4
    head = MLPPredictionHead(in_dim, hidden_dim, out_dim, num_layers)

    assert head.num_layers == num_layers
    assert len(head.layers) == num_layers
    assert all(isinstance(m, torch.nn.Linear) for m in head.layers)


def test_layer_dimensions_match_spec():
    in_dim, hidden_dim, out_dim, num_layers = 10, 32, 4, 3
    head = MLPPredictionHead(in_dim, hidden_dim, out_dim, num_layers)

    # layer0: in_dim -> hidden_dim
    assert head.layers[0].in_features == in_dim
    assert head.layers[0].out_features == hidden_dim

    # middle layers: hidden_dim -> hidden_dim
    for layer in head.layers[1:-1]:
        assert layer.in_features == hidden_dim
        assert layer.out_features == hidden_dim

    # last: hidden_dim -> out_dim
    assert head.layers[-1].in_features == hidden_dim
    assert head.layers[-1].out_features == out_dim
