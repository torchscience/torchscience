"""Tests for ChebyshevT root finding."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t,
    chebyshev_t_companion,
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
