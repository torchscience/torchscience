# tests/torchscience/root_finding/test__brent.py
import math

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
