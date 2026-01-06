"""Tests for ChebyshevT root finding."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_companion,
    chebyshev_t_evaluate,
    chebyshev_t_roots,
)


class TestChebyshevTCompanion:
    """Tests for chebyshev_t_companion."""

    def test_companion_t2(self):
        """Companion matrix for T_2."""
        # T_2 has roots at x = ±sqrt(2)/2
        c = chebyshev_t(torch.tensor([0.0, 0.0, 1.0]))  # T_2
        A = chebyshev_t_companion(c)
        assert A.shape == (2, 2)

    def test_companion_t3(self):
        """Companion matrix for T_3."""
        c = chebyshev_t(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T_3
        A = chebyshev_t_companion(c)
        assert A.shape == (3, 3)

    def test_companion_eigenvalues_are_roots(self):
        """Eigenvalues of companion matrix are roots."""
        # T_3 = 4x^3 - 3x, roots at 0, ±sqrt(3)/2
        c = chebyshev_t(
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        A = chebyshev_t_companion(c)
        eigenvalues = torch.linalg.eigvalsh(A)

        expected_roots = torch.tensor(
            [0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2], dtype=torch.float64
        )
        expected_roots = expected_roots.sort().values
        eigenvalues = eigenvalues.sort().values

        torch.testing.assert_close(
            eigenvalues, expected_roots, atol=1e-10, rtol=1e-10
        )

    def test_companion_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebcompanion."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 1.0]  # T_4

        c = chebyshev_t(torch.tensor(coeffs, dtype=torch.float64))
        A_torch = chebyshev_t_companion(c)

        A_np = np_cheb.chebcompanion(coeffs)

        np.testing.assert_allclose(A_torch.numpy(), A_np, rtol=1e-10)


class TestChebyshevTRoots:
    """Tests for chebyshev_t_roots."""

    def test_roots_t2(self):
        """Roots of T_2."""
        # T_2 = 2x^2 - 1, roots at ±sqrt(2)/2
        c = chebyshev_t(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        roots = chebyshev_t_roots(c)

        expected = torch.tensor(
            [np.sqrt(2) / 2, -np.sqrt(2) / 2], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_t3(self):
        """Roots of T_3."""
        # T_3 = 4x^3 - 3x, roots at 0, ±sqrt(3)/2
        c = chebyshev_t(
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        )
        roots = chebyshev_t_roots(c)

        expected = torch.tensor(
            [0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2], dtype=torch.float64
        )
        expected = expected.sort().values
        roots_sorted = roots.real.sort().values

        torch.testing.assert_close(
            roots_sorted, expected, atol=1e-10, rtol=1e-10
        )

    def test_roots_verify_evaluation(self):
        """f(root) ≈ 0 for all roots."""
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0], dtype=torch.float64)
        c = chebyshev_t(coeffs)
        roots = chebyshev_t_roots(c)

        # Evaluate at roots
        f_at_roots = chebyshev_t_evaluate(c, roots)

        torch.testing.assert_close(
            f_at_roots.abs(),
            torch.zeros_like(f_at_roots.abs()),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_roots_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebroots."""
        coeffs = [1.0, -2.0, 3.0, -1.0, 1.0]

        c = chebyshev_t(torch.tensor(coeffs, dtype=torch.float64))
        roots_torch = chebyshev_t_roots(c)

        roots_np = np_cheb.chebroots(coeffs)

        # Sort for comparison
        roots_torch_sorted = roots_torch.real.sort().values.numpy()
        roots_np_sorted = np.sort(roots_np.real)

        np.testing.assert_allclose(
            roots_torch_sorted, roots_np_sorted, rtol=1e-8
        )
