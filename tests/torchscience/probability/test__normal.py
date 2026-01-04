import math

import scipy.stats
import torch

import torchscience._csrc  # noqa: F401 - Load C++ operators


class TestNormalCdfForward:
    """Test normal_cdf forward correctness."""

    def test_standard_normal(self):
        """Test standard normal CDF at key quantiles."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = torch.ops.torchscience.normal_cdf(
            x, torch.tensor(0.0), torch.tensor(1.0)
        )
        expected = torch.tensor([0.0228, 0.1587, 0.5000, 0.8413, 0.9772])
        assert torch.allclose(result, expected, atol=1e-4)

    def test_scipy_comparison(self):
        """Compare against scipy.stats.norm.cdf."""
        x = torch.linspace(-4, 4, 100)
        loc = torch.tensor(1.5)
        scale = torch.tensor(2.0)

        result = torch.ops.torchscience.normal_cdf(x, loc, scale)
        expected = torch.tensor(
            scipy.stats.norm.cdf(x.numpy(), loc=1.5, scale=2.0),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-6)

    def test_symmetry(self):
        """CDF(mu - x) + CDF(mu + x) = 1."""
        x = torch.linspace(0.1, 3, 50)
        mu = torch.tensor(2.0)
        sigma = torch.tensor(1.5)

        cdf_minus = torch.ops.torchscience.normal_cdf(mu - x, mu, sigma)
        cdf_plus = torch.ops.torchscience.normal_cdf(mu + x, mu, sigma)

        assert torch.allclose(
            cdf_minus + cdf_plus, torch.ones_like(x), atol=1e-6
        )


class TestNormalCdfMeta:
    """Test meta tensor support."""

    def test_meta_tensor_shape(self):
        """Meta tensors should produce correct output shape."""
        x = torch.randn(3, 4, device="meta")
        loc = torch.randn(3, 4, device="meta")
        scale = torch.randn(3, 4, device="meta")
        result = torch.ops.torchscience.normal_cdf(x, loc, scale)
        assert result.shape == (3, 4)
        assert result.device.type == "meta"

    def test_meta_tensor_broadcast(self):
        """Meta tensors should broadcast correctly."""
        x = torch.randn(3, 4, device="meta")
        loc = torch.randn(1, device="meta")
        scale = torch.randn(1, device="meta")
        result = torch.ops.torchscience.normal_cdf(x, loc, scale)
        assert result.shape == (3, 4)


class TestNormalCdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.normal_cdf(x_, loc, scale)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_loc(self):
        """Gradient check for loc parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        scale = torch.tensor(1.0, dtype=torch.float64)

        def fn(loc_):
            return torch.ops.torchscience.normal_cdf(x, loc_, scale)

        assert torch.autograd.gradcheck(fn, (loc,), eps=1e-6, atol=1e-4)

    def test_gradcheck_scale(self):
        """Gradient check for scale parameter."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(scale_):
            return torch.ops.torchscience.normal_cdf(x, loc, scale_)

        assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-4)

    def test_grad_x_is_pdf(self):
        """dCDF/dx should equal PDF."""
        x = torch.linspace(-3, 3, 100, dtype=torch.float64, requires_grad=True)
        loc = torch.tensor(0.0, dtype=torch.float64)
        scale = torch.tensor(1.0, dtype=torch.float64)

        cdf = torch.ops.torchscience.normal_cdf(x, loc, scale)
        grad_x = torch.autograd.grad(cdf.sum(), x)[0]

        # PDF = exp(-z^2/2) / (sigma * sqrt(2*pi))
        z = (x - loc) / scale
        pdf = torch.exp(-0.5 * z * z) / (scale * math.sqrt(2 * math.pi))

        assert torch.allclose(grad_x, pdf.detach(), atol=1e-6)
