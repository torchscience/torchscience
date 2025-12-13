import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import complete_elliptic_integral_k


class TestCompleteEllipticIntegralK(UnaryOperatorTestCase):
    func = staticmethod(complete_elliptic_integral_k)
    op_name = "_complete_elliptic_integral_k"

    symmetry = "even"  # K(k) = K(-k)
    period = None
    bounds = None  # K diverges as k -> 1

    known_values = {
        0.0: math.pi / 2,  # K(0) = pi/2
    }

    reference = staticmethod(lambda k: torch.from_numpy(
        scipy.special.ellipk(k.numpy())
    ).to(k.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-0.99, 0.99)  # Avoid singularity at |k| = 1
    gradcheck_inputs = [0.0, 0.3, 0.5, 0.7, 0.9]
    extreme_values = [0.0, 0.5, 0.9, 0.99]

    def test_at_zero(self):
        """Test K(0) = pi/2."""
        k = torch.tensor([0.0], dtype=torch.float64)
        result = complete_elliptic_integral_k(k)
        expected = torch.tensor([math.pi / 2], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_symmetry(self):
        """Test K(k) = K(-k)."""
        k = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        k_pos = complete_elliptic_integral_k(k)
        k_neg = complete_elliptic_integral_k(-k)
        torch.testing.assert_close(k_pos, k_neg, atol=1e-6, rtol=1e-6)

    def test_increasing(self):
        """Test K(k) is increasing for k in [0, 1)."""
        k = torch.linspace(0.0, 0.95, 20)
        result = complete_elliptic_integral_k(k)
        diff = result[1:] - result[:-1]
        assert torch.all(diff >= -1e-6), "K should be increasing"

    def test_divergence_near_one(self):
        """Test K(k) -> infinity as k -> 1."""
        k_values = torch.tensor([0.9, 0.99, 0.999], dtype=torch.float64)
        result = complete_elliptic_integral_k(k_values)
        # Values should be increasing rapidly
        assert result[1] > result[0], "K should increase as k approaches 1"
        assert result[2] > result[1], "K should diverge"

    def test_specific_values(self):
        """Test specific known values."""
        test_cases = [
            (0.0, math.pi / 2),
            (0.5, 1.6857503548),  # K(0.5)
            (0.9, 2.5780921134),  # K(0.9)
        ]
        for k_val, expected_val in test_cases:
            k = torch.tensor([k_val], dtype=torch.float64)
            result = complete_elliptic_integral_k(k)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_lower_bound(self):
        """Test K(k) >= pi/2 for all k in [0, 1)."""
        k = torch.linspace(0.0, 0.99, 50)
        result = complete_elliptic_integral_k(k)
        assert torch.all(result >= math.pi / 2 - 1e-6), "K should be >= pi/2"
