"""Tests for ChebyshevT tensorclass."""

import pytest
import torch

from torchscience.polynomial import chebyshev_t


class TestChebyshevTConstructor:
    """Tests for chebyshev_t() constructor."""

    def test_single_coefficient(self):
        """Constant Chebyshev series."""
        c = chebyshev_t(torch.tensor([3.0]))
        assert c.coeffs.shape == (1,)
        assert c.coeffs[0] == 3.0

    def test_multiple_coefficients(self):
        """Standard Chebyshev series."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        assert c.coeffs.shape == (3,)
        torch.testing.assert_close(c.coeffs, torch.tensor([1.0, 2.0, 3.0]))

    def test_empty_raises(self):
        """Empty coefficients raise error."""
        from torchscience.polynomial import PolynomialError

        with pytest.raises(PolynomialError):
            chebyshev_t(torch.tensor([]))

    def test_preserves_dtype(self):
        """Dtype is preserved."""
        c = chebyshev_t(torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert c.coeffs.dtype == torch.float64

    def test_preserves_device(self):
        """Device is preserved."""
        coeffs = torch.tensor([1.0, 2.0])
        c = chebyshev_t(coeffs)
        assert c.coeffs.device == coeffs.device


from torchscience.polynomial import chebyshev_t_evaluate


class TestChebyshevTEvaluate:
    """Tests for chebyshev_t_evaluate using Clenshaw algorithm."""

    def test_evaluate_constant(self):
        """T_0(x) = 1 for all x."""
        c = chebyshev_t(torch.tensor([3.0]))
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        y = chebyshev_t_evaluate(c, x)
        torch.testing.assert_close(y, torch.tensor([3.0, 3.0, 3.0, 3.0]))

    def test_evaluate_t1(self):
        """T_1(x) = x."""
        c = chebyshev_t(torch.tensor([0.0, 1.0]))  # T_1
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        y = chebyshev_t_evaluate(c, x)
        torch.testing.assert_close(y, x)

    def test_evaluate_t2(self):
        """T_2(x) = 2x^2 - 1."""
        c = chebyshev_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        expected = 2 * x**2 - 1  # [-1+2, -1, -0.5, 1] = [1, -1, -0.5, 1]
        y = chebyshev_t_evaluate(c, x)
        torch.testing.assert_close(y, expected)

    def test_evaluate_linear_combination(self):
        """1 + 2*T_1 + 3*T_2 at x=0.5."""
        # T_0(0.5) = 1, T_1(0.5) = 0.5, T_2(0.5) = 2*0.25 - 1 = -0.5
        # Result: 1 + 2*0.5 + 3*(-0.5) = 1 + 1 - 1.5 = 0.5
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([0.5])
        y = chebyshev_t_evaluate(c, x)
        torch.testing.assert_close(y, torch.tensor([0.5]))

    def test_evaluate_at_chebyshev_points(self):
        """T_n(cos(k*pi/n)) = cos(k*pi)."""
        import math

        # T_3 at cos(k*pi/3) for k=0,1,2,3
        c = chebyshev_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        k = torch.tensor([0.0, 1.0, 2.0, 3.0])
        x = torch.cos(k * math.pi / 3)
        y = chebyshev_t_evaluate(c, x)
        expected = torch.cos(k * math.pi)  # [1, -1, 1, -1]
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_call_operator(self):
        """Test __call__ operator."""
        c = chebyshev_t(torch.tensor([1.0, 2.0]))
        x = torch.tensor([0.0, 1.0])
        y = c(x)
        # T_0=1, T_1=x: 1 + 2*0=1, 1 + 2*1=3
        torch.testing.assert_close(y, torch.tensor([1.0, 3.0]))
