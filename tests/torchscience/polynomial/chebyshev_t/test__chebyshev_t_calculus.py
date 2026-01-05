"""Tests for ChebyshevT calculus operations."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_antiderivative,
    chebyshev_t_derivative,
    chebyshev_t_evaluate,
)


class TestChebyshevTDerivative:
    """Tests for chebyshev_t_derivative."""

    def test_derivative_constant(self):
        """Derivative of constant is zero."""
        a = chebyshev_t(torch.tensor([5.0]))
        da = chebyshev_t_derivative(a)
        torch.testing.assert_close(da.coeffs, torch.tensor([0.0]))

    def test_derivative_t1(self):
        """d/dx T_1(x) = 1 = T_0."""
        a = chebyshev_t(torch.tensor([0.0, 1.0]))  # T_1 = x
        da = chebyshev_t_derivative(a)
        torch.testing.assert_close(da.coeffs, torch.tensor([1.0]))

    def test_derivative_t2(self):
        """d/dx T_2(x) = 4x = 4*T_1."""
        # T_2 = 2x^2 - 1, d/dx = 4x = 4*T_1
        a = chebyshev_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        da = chebyshev_t_derivative(a)
        torch.testing.assert_close(da.coeffs, torch.tensor([0.0, 4.0]))

    def test_derivative_t3(self):
        """d/dx T_3(x) = 3*U_2(x) = 12x^2 - 3 = 6*T_2 + 3*T_0."""
        # T_3 = 4x^3 - 3x, d/dx = 12x^2 - 3
        # In Chebyshev basis: 12x^2 - 3 = 12*(T_2+1)/2 - 3 = 6*T_2 + 6 - 3 = 3 + 6*T_2
        # = 3*T_0 + 6*T_2
        a = chebyshev_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        da = chebyshev_t_derivative(a)
        torch.testing.assert_close(da.coeffs, torch.tensor([3.0, 0.0, 6.0]))

    def test_derivative_linear_combination(self):
        """Derivative of 1 + 2*T_1 + 3*T_2."""
        # d/dx (1 + 2*T_1 + 3*T_2) = 2*1 + 3*4*T_1 = 2 + 12*T_1
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        da = chebyshev_t_derivative(a)
        torch.testing.assert_close(da.coeffs, torch.tensor([2.0, 12.0]))

    def test_derivative_second_order(self):
        """Second derivative."""
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # degree 3
        d2a = chebyshev_t_derivative(a, order=2)
        # First derivative: degree 2
        # Second derivative: degree 1
        assert d2a.coeffs.shape[-1] <= 3

    def test_derivative_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebder."""
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]

        a = chebyshev_t(torch.tensor(coeffs))
        da = chebyshev_t_derivative(a)

        da_np = np_cheb.chebder(coeffs)

        np.testing.assert_allclose(da.coeffs.numpy(), da_np, rtol=1e-6)

    def test_derivative_evaluation_consistency(self):
        """Numerical derivative matches symbolic derivative."""
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        da = chebyshev_t_derivative(a)

        x = torch.tensor([0.0, 0.3, 0.7], requires_grad=True)
        y = chebyshev_t_evaluate(a, x)

        # Compute numerical derivative
        grad_y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

        # Evaluate symbolic derivative
        dy_symbolic = chebyshev_t_evaluate(da, x.detach())

        torch.testing.assert_close(grad_y, dy_symbolic, atol=1e-5, rtol=1e-5)


class TestChebyshevTAntiderivative:
    """Tests for chebyshev_t_antiderivative."""

    def test_antiderivative_constant(self):
        """Antiderivative of constant."""
        # integral(1) = T_1 = x (with C=0)
        a = chebyshev_t(torch.tensor([1.0]))  # 1 = T_0
        ia = chebyshev_t_antiderivative(a)
        # Result: 0 + T_1 = [0, 1]
        torch.testing.assert_close(ia.coeffs, torch.tensor([0.0, 1.0]))

    def test_antiderivative_with_constant(self):
        """Antiderivative with integration constant."""
        a = chebyshev_t(torch.tensor([1.0]))  # constant 1
        ia = chebyshev_t_antiderivative(a, constant=2.0)
        # integral(1) + 2 = x + 2 = 2*T_0 + T_1
        torch.testing.assert_close(ia.coeffs, torch.tensor([2.0, 1.0]))

    def test_antiderivative_derivative_inverse(self):
        """Derivative of antiderivative recovers original."""
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        ia = chebyshev_t_antiderivative(a)
        dia = chebyshev_t_derivative(ia)
        torch.testing.assert_close(dia.coeffs, a.coeffs, atol=1e-6, rtol=1e-6)

    def test_antiderivative_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebint."""
        coeffs = [1.0, 2.0, 3.0]

        a = chebyshev_t(torch.tensor(coeffs))
        ia = chebyshev_t_antiderivative(a, constant=0.0)

        ia_np = np_cheb.chebint(coeffs)

        np.testing.assert_allclose(ia.coeffs.numpy(), ia_np, rtol=1e-6)

    def test_antiderivative_order_2(self):
        """Second antiderivative."""
        coeffs = [1.0, 2.0, 3.0]

        a = chebyshev_t(torch.tensor(coeffs))
        i2a = chebyshev_t_antiderivative(a, order=2)

        i2a_np = np_cheb.chebint(coeffs, m=2)

        np.testing.assert_allclose(i2a.coeffs.numpy(), i2a_np, rtol=1e-5)

    def test_antiderivative_t1_vs_numpy(self):
        """Antiderivative of T_1 matches numpy."""
        coeffs = [0.0, 1.0]  # T_1

        a = chebyshev_t(torch.tensor(coeffs))
        ia = chebyshev_t_antiderivative(a, constant=0.0)

        ia_np = np_cheb.chebint(coeffs)

        np.testing.assert_allclose(ia.coeffs.numpy(), ia_np, rtol=1e-6)
