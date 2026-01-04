import pytest
import scipy.integrate
import torch


class TestSimpson:
    def test_matches_scipy(self):
        """Compare with scipy.integrate.simpson"""
        x = torch.linspace(0, torch.pi, 101)  # Odd number of points
        y = torch.sin(x)

        from torchscience.integration.quadrature import simpson

        result = simpson(y, x)

        expected = scipy.integrate.simpson(y.numpy(), x=x.numpy())
        assert torch.allclose(
            result, torch.tensor(expected, dtype=result.dtype), rtol=1e-6
        )

    def test_higher_order_accuracy(self):
        """Simpson should be more accurate than trapezoid for smooth functions"""
        x = torch.linspace(0, 1, 11)
        y = x**4  # integral from 0 to 1 = 1/5 = 0.2

        from torchscience.integration.quadrature import simpson, trapezoid

        trap_result = trapezoid(y, x)
        simp_result = simpson(y, x)

        exact = 0.2
        trap_error = abs(trap_result.item() - exact)
        simp_error = abs(simp_result.item() - exact)

        assert simp_error < trap_error

    def test_requires_at_least_3_points(self):
        """Simpson's rule needs at least 3 points"""
        y = torch.tensor([1.0, 2.0])

        from torchscience.integration.quadrature import simpson

        with pytest.raises(ValueError, match="at least 3"):
            simpson(y)


class TestSimpsonEven:
    def test_even_avg(self):
        """Test even='avg' handling for even number of intervals"""
        x = torch.linspace(0, 1, 10)  # 9 intervals (odd, needs handling)
        y = x**2

        from torchscience.integration.quadrature import simpson

        result = simpson(y, x, even="avg")

        # Should be close to 1/3
        assert torch.allclose(result, torch.tensor(1 / 3), rtol=1e-2)

    def test_even_first(self):
        """Test even='first' handling"""
        x = torch.linspace(0, 1, 10)
        y = x**2

        from torchscience.integration.quadrature import simpson

        result = simpson(y, x, even="first")

        assert torch.allclose(result, torch.tensor(1 / 3), rtol=1e-2)

    def test_invalid_even_raises(self):
        """Invalid even parameter should raise"""
        x = torch.linspace(0, 1, 10)
        y = x**2

        from torchscience.integration.quadrature import simpson

        with pytest.raises(ValueError, match="even must be"):
            simpson(y, x, even="invalid")


class TestSimpsonGradients:
    def test_gradcheck(self):
        """Numerical gradient check"""
        x = torch.linspace(0, 1, 11, dtype=torch.float64)
        y = torch.randn(11, requires_grad=True, dtype=torch.float64)

        from torchscience.integration.quadrature import simpson

        def fn(y_):
            return simpson(y_, x)

        assert torch.autograd.gradcheck(fn, (y,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient check"""
        x = torch.linspace(0, 1, 11, dtype=torch.float64)
        y = torch.randn(11, requires_grad=True, dtype=torch.float64)

        from torchscience.integration.quadrature import simpson

        def fn(y_):
            return simpson(y_, x)

        assert torch.autograd.gradgradcheck(fn, (y,), raise_exception=True)


class TestSimpsonBatching:
    def test_batch_dimension(self):
        """Integrate over batched data"""
        x = torch.linspace(0, 1, 101)
        y = torch.stack([x**2, x**3, x**4], dim=0)  # Shape: (3, 101)

        from torchscience.integration.quadrature import simpson

        result = simpson(y, x, dim=-1)

        assert result.shape == (3,)
        assert torch.allclose(result[0], torch.tensor(1 / 3), rtol=1e-4)
        assert torch.allclose(result[1], torch.tensor(1 / 4), rtol=1e-4)
        assert torch.allclose(result[2], torch.tensor(1 / 5), rtol=1e-4)
