import pytest
import scipy.stats
import torch

import torchscience  # noqa: F401 - loads the C++ extension


class TestChi2CdfForward:
    """Test chi2_cdf forward correctness."""

    def test_scipy_comparison(self):
        """Compare against scipy.stats.chi2.cdf."""
        x = torch.linspace(0.1, 20, 100)
        df = torch.tensor(5.0)

        result = torch.ops.torchscience.chi2_cdf(x, df)
        expected = torch.tensor(
            scipy.stats.chi2.cdf(x.numpy(), df=5),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_at_zero(self):
        """CDF(0) = 0."""
        x = torch.tensor([0.0])
        df = torch.tensor(5.0)
        result = torch.ops.torchscience.chi2_cdf(x, df)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_chi2_2_is_exponential(self):
        """Chi2(2) = Exponential(1/2), so CDF = 1 - exp(-x/2)."""
        x = torch.linspace(0.1, 10, 50)
        df = torch.tensor(2.0)

        result = torch.ops.torchscience.chi2_cdf(x, df)
        expected = 1 - torch.exp(-x / 2)

        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize("df", [1.0, 2.0, 5.0, 10.0, 30.0])
    def test_various_df(self, df):
        """Test various degrees of freedom."""
        x = torch.linspace(0.1, 30, 50)
        df_t = torch.tensor(df)

        result = torch.ops.torchscience.chi2_cdf(x, df_t)
        expected = torch.tensor(
            scipy.stats.chi2.cdf(x.numpy(), df=df),
            dtype=torch.float32,
        )
        assert torch.allclose(result, expected, atol=1e-5)


class TestChi2CdfGradients:
    """Test gradient computation."""

    def test_gradcheck_x(self):
        """Gradient check for x parameter."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64)

        def fn(x_):
            return torch.ops.torchscience.chi2_cdf(x_, df)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradcheck_df(self):
        """Gradient check for df parameter."""
        x = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(df_):
            return torch.ops.torchscience.chi2_cdf(x, df_)

        assert torch.autograd.gradcheck(fn, (df,), eps=1e-6, atol=1e-4)

    def test_gradcheck_both(self):
        """Gradient check for both parameters."""
        x = torch.tensor(
            [1.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True
        )
        df = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)

        def fn(x_, df_):
            return torch.ops.torchscience.chi2_cdf(x_, df_)

        assert torch.autograd.gradcheck(fn, (x, df), eps=1e-6, atol=1e-4)
