"""Tests for ChebyshevT fitting and interpolation."""

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_t_points,
)


class TestChebyshevTPoints:
    """Tests for chebyshev_t_points (Chebyshev nodes)."""

    def test_points_n1(self):
        """Single Chebyshev point."""
        x = chebyshev_t_points(1)
        # x_0 = cos(pi/2) = 0
        torch.testing.assert_close(x, torch.tensor([0.0]))

    def test_points_n2(self):
        """Two Chebyshev points."""
        x = chebyshev_t_points(2)
        # x_k = cos((2k+1)*pi/(2n)) for k=0,1
        # x_0 = cos(pi/4) = sqrt(2)/2
        # x_1 = cos(3*pi/4) = -sqrt(2)/2
        expected = torch.tensor(
            [np.sqrt(2) / 2, -np.sqrt(2) / 2], dtype=torch.float32
        )
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_points_n5(self):
        """Five Chebyshev points."""
        x = chebyshev_t_points(5)
        assert x.shape == (5,)
        # Points should be in [-1, 1]
        assert x.min() >= -1.0
        assert x.max() <= 1.0
        # Points should be symmetric around 0
        torch.testing.assert_close(
            x.sort().values + x.sort(descending=True).values,
            torch.zeros(5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_points_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebpts1."""
        n = 10
        x_torch = chebyshev_t_points(n)
        x_np = np_cheb.chebpts1(n)
        # NumPy uses ascending order, ours is descending, so flip for comparison
        np.testing.assert_allclose(x_torch.numpy()[::-1], x_np, rtol=1e-6)

    def test_points_dtype(self):
        """Preserve dtype."""
        x = chebyshev_t_points(5, dtype=torch.float64)
        assert x.dtype == torch.float64
