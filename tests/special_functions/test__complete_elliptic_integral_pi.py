import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import complete_elliptic_integral_pi, complete_elliptic_integral_k


class TestCompleteEllipticIntegralPi(BinaryOperatorTestCase):
    func = staticmethod(complete_elliptic_integral_pi)
    op_name = "_complete_elliptic_integral_pi"

    known_values = []

    reference = None  # scipy uses different conventions

    input_range_1 = (-0.9, 0.9)  # n (characteristic)
    input_range_2 = (-0.9, 0.9)  # k (modulus)

    gradcheck_inputs = ([0.0, 0.3, 0.5], [0.0, 0.3, 0.5])

    supports_complex = False

    def test_n_zero(self):
        """Test Pi(0, k) = K(k)."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.5, 0.8], dtype=torch.float64)
        result = complete_elliptic_integral_pi(n, k)
        expected = complete_elliptic_integral_k(k)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_at_zero_zero(self):
        """Test Pi(0, 0) = pi/2."""
        n = torch.tensor([0.0], dtype=torch.float64)
        k = torch.tensor([0.0], dtype=torch.float64)
        result = complete_elliptic_integral_pi(n, k)
        expected = torch.tensor([math.pi / 2], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_positive_for_small_n(self):
        """Test Pi(n, k) > 0 for small n and |k| < 1."""
        n = torch.linspace(-0.5, 0.5, 10)
        k = torch.linspace(0.0, 0.5, 10)
        result = complete_elliptic_integral_pi(n, k)
        assert torch.all(result > 0), "Pi should be positive for small n"
