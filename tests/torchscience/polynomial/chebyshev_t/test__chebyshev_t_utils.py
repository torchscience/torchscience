"""Tests for ChebyshevT utility functions."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_degree,
    chebyshev_t_trim,
)


class TestChebyshevTDegree:
    """Tests for chebyshev_t_degree."""

    def test_degree_constant(self):
        """Degree of constant is 0."""
        c = chebyshev_t(torch.tensor([5.0]))
        assert chebyshev_t_degree(c).item() == 0

    def test_degree_linear(self):
        """Degree of linear is 1."""
        c = chebyshev_t(torch.tensor([1.0, 2.0]))
        assert chebyshev_t_degree(c).item() == 1

    def test_degree_quadratic(self):
        """Degree of quadratic is 2."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        assert chebyshev_t_degree(c).item() == 2

    def test_degree_with_trailing_zeros(self):
        """Degree counts all coefficients."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 0.0]))
        assert chebyshev_t_degree(c).item() == 2  # Still degree 2


class TestChebyshevTTrim:
    """Tests for chebyshev_t_trim."""

    def test_trim_no_change(self):
        """Trim doesn't change non-zero trailing."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 3.0]))
        t = chebyshev_t_trim(c)
        torch.testing.assert_close(t.coeffs, c.coeffs)

    def test_trim_trailing_zeros(self):
        """Trim removes trailing zeros."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 0.0, 0.0]))
        t = chebyshev_t_trim(c)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_near_zero(self):
        """Trim removes near-zero trailing coefficients."""
        c = chebyshev_t(torch.tensor([1.0, 2.0, 1e-15, 1e-16]))
        t = chebyshev_t_trim(c, tol=1e-10)
        torch.testing.assert_close(t.coeffs, torch.tensor([1.0, 2.0]))

    def test_trim_preserves_constant(self):
        """Trim preserves at least one coefficient."""
        c = chebyshev_t(torch.tensor([0.0, 0.0, 0.0]))
        t = chebyshev_t_trim(c)
        assert t.coeffs.shape[-1] >= 1

    def test_trim_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebtrim."""
        coeffs = [1.0, 2.0, 0.0, 1e-16, 0.0]

        c = chebyshev_t(torch.tensor(coeffs))
        t = chebyshev_t_trim(c, tol=1e-10)

        t_np = np_cheb.chebtrim(coeffs, tol=1e-10)

        np.testing.assert_allclose(t.coeffs.numpy(), t_np, rtol=1e-6)
