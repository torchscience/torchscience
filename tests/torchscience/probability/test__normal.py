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
