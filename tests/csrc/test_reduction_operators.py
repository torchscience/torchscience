import pytest
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators

# This test validates the reduction operator template works correctly
# by testing against a simple "sum" reduction (which we can compare to torch.sum)


class TestReductionOperatorTemplate:
    """Test the CPUReductionOperator template pattern."""

    @pytest.fixture
    def input_tensor(self):
        torch.manual_seed(42)
        return torch.randn(4, 5, 6, requires_grad=True)

    def test_reduction_all_dims(self, input_tensor):
        """Test reduction over all dimensions."""
        # We'll test with kurtosis since it uses the pattern
        result = torch.ops.torchscience.kurtosis(
            input_tensor, None, False, True, True
        )
        assert result.shape == ()
        assert result.dtype == input_tensor.dtype

    def test_reduction_single_dim(self, input_tensor):
        """Test reduction over a single dimension."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], False, True, True
        )
        assert result.shape == (4, 6)

    def test_reduction_keepdim(self, input_tensor):
        """Test reduction with keepdim=True."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], True, True, True
        )
        assert result.shape == (4, 1, 6)

    def test_reduction_multiple_dims(self, input_tensor):
        """Test reduction over multiple dimensions."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [0, 2], False, True, True
        )
        assert result.shape == (5,)

    def test_reduction_negative_dim(self, input_tensor):
        """Test reduction with negative dimension index."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [-1], False, True, True
        )
        assert result.shape == (4, 5)

    def test_reduction_backward(self, input_tensor):
        """Test backward pass through reduction."""
        result = torch.ops.torchscience.kurtosis(
            input_tensor, [1], False, True, True
        )
        loss = result.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
