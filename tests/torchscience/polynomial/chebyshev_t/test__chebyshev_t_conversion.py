"""Tests for ChebyshevT basis conversion."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_evaluate,
    chebyshev_t_to_polynomial,
    polynomial_evaluate,
)


class TestChebyshevTToPolynomial:
    """Tests for chebyshev_t_to_polynomial."""

    def test_t0_to_power(self):
        """T_0 = 1."""
        c = chebyshev_t(torch.tensor([1.0]))  # T_0
        p = chebyshev_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([1.0]))

    def test_t1_to_power(self):
        """T_1 = x."""
        c = chebyshev_t(torch.tensor([0.0, 1.0]))  # T_1
        p = chebyshev_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([0.0, 1.0]))

    def test_t2_to_power(self):
        """T_2 = 2x^2 - 1."""
        c = chebyshev_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        p = chebyshev_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([-1.0, 0.0, 2.0]))

    def test_t3_to_power(self):
        """T_3 = 4x^3 - 3x."""
        c = chebyshev_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        p = chebyshev_t_to_polynomial(c)
        torch.testing.assert_close(
            p.coeffs, torch.tensor([0.0, -3.0, 0.0, 4.0])
        )

    def test_linear_combination(self):
        """1 + 2*T_1 + 3*T_2 in power basis."""
        # = 1 + 2x + 3*(2x^2 - 1) = 1 + 2x + 6x^2 - 3 = -2 + 2x + 6x^2
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        p = chebyshev_t_to_polynomial(c)
        torch.testing.assert_close(p.coeffs, torch.tensor([-2.0, 2.0, 6.0]))

    def test_evaluation_consistency(self):
        """Chebyshev and power give same values."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        c = chebyshev_t(coeffs)
        p = chebyshev_t_to_polynomial(c)

        x = torch.linspace(-1, 1, 20)
        y_cheb = chebyshev_t_evaluate(c, x)
        y_power = polynomial_evaluate(p, x)

        torch.testing.assert_close(y_cheb, y_power, atol=1e-5, rtol=1e-5)

    def test_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.cheb2poly."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        c = chebyshev_t(torch.tensor(coeffs, dtype=torch.float64))
        p = chebyshev_t_to_polynomial(c)

        p_np = np_cheb.cheb2poly(coeffs)

        np.testing.assert_allclose(p.coeffs.numpy(), p_np, rtol=1e-10)
