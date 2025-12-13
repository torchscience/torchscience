import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import complete_elliptic_integral_e


class TestCompleteEllipticIntegralE(UnaryOperatorTestCase):
    func = staticmethod(complete_elliptic_integral_e)
    op_name = "_complete_elliptic_integral_e"

    symmetry = "even"  # E(k) = E(-k)
    period = None
    bounds = (1.0, math.pi / 2)  # E is bounded

    known_values = {
        0.0: math.pi / 2,  # E(0) = pi/2
        1.0: 1.0,          # E(1) = 1
    }

    reference = staticmethod(lambda k: torch.from_numpy(
        scipy.special.ellipe(k.numpy())
    ).to(k.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-0.99, 0.99)
    gradcheck_inputs = [0.0, 0.3, 0.5, 0.7, 0.9]
    extreme_values = [0.0, 0.5, 0.9, 0.99]

    def test_at_zero(self):
        """Test E(0) = pi/2."""
        k = torch.tensor([0.0], dtype=torch.float64)
        result = complete_elliptic_integral_e(k)
        expected = torch.tensor([math.pi / 2], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_at_one(self):
        """Test E(1) = 1."""
        k = torch.tensor([1.0], dtype=torch.float64)
        result = complete_elliptic_integral_e(k)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_symmetry(self):
        """Test E(k) = E(-k)."""
        k = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        e_pos = complete_elliptic_integral_e(k)
        e_neg = complete_elliptic_integral_e(-k)
        torch.testing.assert_close(e_pos, e_neg, atol=1e-6, rtol=1e-6)

    def test_decreasing(self):
        """Test E(k) is decreasing for k in [0, 1]."""
        k = torch.linspace(0.0, 0.95, 20)
        result = complete_elliptic_integral_e(k)
        diff = result[1:] - result[:-1]
        assert torch.all(diff <= 1e-6), "E should be decreasing"

    def test_bounded(self):
        """Test 1 <= E(k) <= pi/2 for k in [0, 1]."""
        k = torch.linspace(0.0, 0.99, 50)
        result = complete_elliptic_integral_e(k)
        assert torch.all(result >= 1.0 - 1e-6), "E should be >= 1"
        assert torch.all(result <= math.pi / 2 + 1e-6), "E should be <= pi/2"

    def test_specific_values(self):
        """Test specific known values."""
        test_cases = [
            (0.0, math.pi / 2),
            (0.5, 1.4674622093),  # E(0.5)
            (0.9, 1.1716970528),  # E(0.9)
        ]
        for k_val, expected_val in test_cases:
            k = torch.tensor([k_val], dtype=torch.float64)
            result = complete_elliptic_integral_e(k)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
