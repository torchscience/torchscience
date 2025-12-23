# tests/torchscience/root_finding/test__brent.py
import math

import pytest
import torch

from torchscience.root_finding import brent


class TestBrent:
    """Tests for Brent's root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )

    def test_batched_roots(self):
        """Find multiple roots in parallel."""
        c = torch.tensor([2.0, 3.0, 4.0, 5.0])
        f = lambda x: x**2 - c
        a = torch.ones(4)
        b = torch.full((4,), 10.0)

        roots = brent(f, a, b)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    def test_trigonometric(self):
        """Find root of sin(x) = 0 in [2, 4] -> pi."""
        f = lambda x: torch.sin(x)
        a = torch.tensor([2.0])
        b = torch.tensor([4.0])

        root = brent(f, a, b)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6
        )

    def test_root_at_endpoint_a(self):
        """Return a immediately if f(a) == 0."""
        f = lambda x: x - 1.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        torch.testing.assert_close(root, torch.tensor([1.0]))

    def test_root_at_endpoint_b(self):
        """Return b immediately if f(b) == 0."""
        f = lambda x: x - 2.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root = brent(f, a, b)

        torch.testing.assert_close(root, torch.tensor([2.0]))

    def test_invalid_bracket_raises(self):
        """Raise ValueError when f(a) and f(b) have same sign."""
        f = lambda x: x**2 + 1  # Always positive
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        with pytest.raises(ValueError, match="Invalid bracket"):
            brent(f, a, b)

    def test_invalid_bracket_count(self):
        """Error message includes count of invalid brackets."""
        f = lambda x: x**2 - torch.tensor([2.0, -1.0, 3.0])  # 2nd is invalid
        a = torch.tensor([1.0, 1.0, 1.0])
        b = torch.tensor([2.0, 2.0, 2.0])

        with pytest.raises(ValueError, match="1 of 3 brackets are invalid"):
            brent(f, a, b)

    def test_shape_mismatch_raises(self):
        """Raise ValueError when a and b have different shapes."""
        f = lambda x: x
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0])

        with pytest.raises(ValueError, match="must have same shape"):
            brent(f, a, b)

    def test_nan_input_raises(self):
        """Raise ValueError when inputs contain NaN."""
        f = lambda x: x
        a = torch.tensor([float("nan")])
        b = torch.tensor([1.0])

        with pytest.raises(ValueError, match="must not contain NaN"):
            brent(f, a, b)

    def test_maxiter_exceeded_raises(self):
        """Raise RuntimeError when maxiter is exceeded."""
        f = lambda x: x**3 - x - 1
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        with pytest.raises(RuntimeError, match="failed to converge"):
            brent(f, a, b, maxiter=1)

    def test_float32(self):
        """Works correctly with float32."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([2.0], dtype=torch.float32)

        root = brent(f, a, b)

        assert root.dtype == torch.float32
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_float64(self):
        """Works correctly with float64."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        root = brent(f, a, b)

        assert root.dtype == torch.float64
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_convergence_xtol(self):
        """Verify interval width is within xtol at convergence."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        xtol = 1e-10

        root = brent(f, a, b, xtol=xtol)

        expected = math.sqrt(2)
        assert abs(root.item() - expected) < xtol * 10

    def test_convergence_ftol(self):
        """Verify |f(x)| is within ftol at convergence."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        ftol = 1e-12

        root = brent(f, a, b, ftol=ftol)

        residual = abs(f(root).item())
        assert residual < ftol * 10

    def test_preserves_shape(self):
        """Output has same shape as input."""
        f = lambda x: x**2 - 2
        a = torch.ones(2, 3)
        b = torch.full((2, 3), 2.0)

        root = brent(f, a, b)

        assert root.shape == (2, 3)

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x
        a = torch.tensor([])
        b = torch.tensor([])

        root = brent(f, a, b)

        assert root.shape == (0,)
