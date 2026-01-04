"""Tests for ChebyshevT arithmetic operations."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_add,
    chebyshev_t_subtract,
)


class TestChebyshevTAdd:
    """Tests for chebyshev_t_add."""

    def test_add_same_degree(self):
        """Add series of same degree."""
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_t(torch.tensor([4.0, 5.0, 6.0]))
        c = chebyshev_t_add(a, b)
        torch.testing.assert_close(c.coeffs, torch.tensor([5.0, 7.0, 9.0]))

    def test_add_different_degree(self):
        """Add series of different degrees (zero-pad shorter)."""
        a = chebyshev_t(torch.tensor([1.0, 2.0]))
        b = chebyshev_t(torch.tensor([3.0, 4.0, 5.0]))
        c = chebyshev_t_add(a, b)
        torch.testing.assert_close(c.coeffs, torch.tensor([4.0, 6.0, 5.0]))

    def test_add_operator(self):
        """Test + operator."""
        a = chebyshev_t(torch.tensor([1.0, 2.0]))
        b = chebyshev_t(torch.tensor([3.0, 4.0]))
        c = a + b
        torch.testing.assert_close(c.coeffs, torch.tensor([4.0, 6.0]))

    def test_add_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebadd."""
        a_coeffs = [1.0, 2.0, 3.0]
        b_coeffs = [4.0, 5.0]

        a = chebyshev_t(torch.tensor(a_coeffs))
        b = chebyshev_t(torch.tensor(b_coeffs))
        c = chebyshev_t_add(a, b)

        c_np = np_cheb.chebadd(a_coeffs, b_coeffs)

        np.testing.assert_allclose(c.coeffs.numpy(), c_np, rtol=1e-6)


class TestChebyshevTSubtract:
    """Tests for chebyshev_t_subtract."""

    def test_subtract_same_degree(self):
        """Subtract series of same degree."""
        a = chebyshev_t(torch.tensor([5.0, 7.0, 9.0]))
        b = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        c = chebyshev_t_subtract(a, b)
        torch.testing.assert_close(c.coeffs, torch.tensor([4.0, 5.0, 6.0]))

    def test_subtract_different_degree(self):
        """Subtract series of different degrees."""
        a = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        b = chebyshev_t(torch.tensor([1.0, 2.0]))
        c = chebyshev_t_subtract(a, b)
        torch.testing.assert_close(c.coeffs, torch.tensor([0.0, 0.0, 3.0]))

    def test_subtract_operator(self):
        """Test - operator."""
        a = chebyshev_t(torch.tensor([5.0, 6.0]))
        b = chebyshev_t(torch.tensor([1.0, 2.0]))
        c = a - b
        torch.testing.assert_close(c.coeffs, torch.tensor([4.0, 4.0]))

    def test_subtract_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebsub."""
        a_coeffs = [5.0, 4.0, 3.0]
        b_coeffs = [1.0, 2.0]

        a = chebyshev_t(torch.tensor(a_coeffs))
        b = chebyshev_t(torch.tensor(b_coeffs))
        c = chebyshev_t_subtract(a, b)

        c_np = np_cheb.chebsub(a_coeffs, b_coeffs)

        np.testing.assert_allclose(c.coeffs.numpy(), c_np, rtol=1e-6)
