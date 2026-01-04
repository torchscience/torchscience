import scipy.special
import torch

import torchscience.special_functions  # noqa: F401 - registers operators


class TestRegularizedGammaPForward:
    """Test regularized_gamma_p forward correctness."""

    def test_basic_values(self):
        """Test against scipy.special.gammainc."""
        a = torch.tensor([1.0, 2.0, 3.0, 5.0])
        x = torch.tensor([0.5, 1.0, 2.0, 3.0])

        result = torch.ops.torchscience.regularized_gamma_p(a, x)

        expected = torch.tensor(
            scipy.special.gammainc(a.numpy(), x.numpy()),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)


class TestRegularizedGammaPSpecialCases:
    """Test special values and edge cases."""

    def test_at_zero(self):
        """P(a, 0) = 0 for all a > 0."""
        a = torch.tensor([0.5, 1.0, 2.0, 10.0])
        x = torch.zeros_like(a)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert torch.allclose(result, torch.zeros_like(a))

    def test_exponential_cdf(self):
        """P(1, x) = 1 - exp(-x) is the exponential CDF."""
        x = torch.linspace(0.1, 5.0, 50)
        a = torch.ones_like(x)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        expected = 1 - torch.exp(-x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_x_approaches_one(self):
        """P(a, x) -> 1 as x -> inf."""
        a = torch.tensor([1.0, 2.0, 5.0])
        x = torch.tensor([100.0, 100.0, 100.0])
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert torch.allclose(result, torch.ones_like(a), atol=1e-10)

    def test_range_zero_to_one(self):
        """P(a, x) is always in [0, 1]."""
        a = torch.rand(100) * 10 + 0.1  # a in (0.1, 10.1)
        x = torch.rand(100) * 20  # x in [0, 20)
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert (result >= 0).all() and (result <= 1).all()


class TestRegularizedGammaPMeta:
    """Test meta tensor support."""

    def test_meta_tensor_shape(self):
        """Meta tensors should produce correct output shape."""
        a = torch.randn(3, 4, device="meta")
        x = torch.randn(3, 4, device="meta")
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcast(self):
        """Meta tensors should broadcast correctly."""
        a = torch.randn(3, 1, device="meta")
        x = torch.randn(1, 4, device="meta")
        result = torch.ops.torchscience.regularized_gamma_p(a, x)
        assert result.shape == (3, 4)
