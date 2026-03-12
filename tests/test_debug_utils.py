import pytest
import torch
from torch import nn

from jormungandr.utils.debug_utils import (
    assert_finite_tensor,
    assert_module_gradients_finite,
    assert_module_parameters_finite,
)


def test_assert_finite_tensor_accepts_finite_values():
    assert_finite_tensor("finite", torch.tensor([1.0, 2.0, 3.0]))


def test_assert_finite_tensor_rejects_nan_values():
    with pytest.raises(RuntimeError, match="bad_tensor"):
        assert_finite_tensor("bad_tensor", torch.tensor([1.0, float("nan")]))


def test_assert_module_parameters_finite_rejects_non_finite_parameter():
    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="linear.weight"):
        assert_module_parameters_finite(layer, "linear")


def test_assert_module_gradients_finite_rejects_non_finite_gradient():
    layer = nn.Linear(2, 2)
    output = layer(torch.ones(1, 2)).sum()
    output.backward()

    layer.weight.grad[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="linear.weight"):
        assert_module_gradients_finite(layer, "linear")
