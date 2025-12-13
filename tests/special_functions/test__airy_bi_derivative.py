import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import airy_bi, airy_bi_derivative


class TestAiryBiDerivative(UnaryOperatorTestCase):
    func = staticmethod(airy_bi_derivative)
    op_name = "_airy_bi_derivative"

    symmetry = None
    period = None
    bounds = None

    # Bi'(0) = 3^(1/6) / Gamma(1/3) ≈ 0.4482883573
    known_values = {
        0.0: 0.4482883573538264,
    }

    zeros = []

    reference = None
    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 5.0)  # Bi grows rapidly for positive x
    gradcheck_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    extreme_values = [-10.0, -5.0, 0.0, 2.0, 3.0]

    def test_at_zero(self):
        """Test Bi'(0) = 3^(1/6) / Gamma(1/3)."""
        result = airy_bi_derivative(torch.tensor([0.0], dtype=torch.float64))
        expected = torch.tensor([0.4482883573538264], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_differential_equation(self):
        """Test Bi' is the derivative of Bi via finite differences."""
        x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0], dtype=torch.float64)
        h = 1e-6

        # Numerical derivative of Bi
        bi_plus = airy_bi(x + h)
        bi_minus = airy_bi(x - h)
        numerical_deriv = (bi_plus - bi_minus) / (2 * h)

        # Bi'(x) from the function
        result = airy_bi_derivative(x)

        torch.testing.assert_close(result, numerical_deriv, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test Bi' at specific known values."""
        test_cases = [
            (0.0, 0.4482883573538264),
            (1.0, 0.9324359333927756),
            (-1.0, 0.5923756264227923),
            (-2.0, -0.27879516692116946),
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = airy_bi_derivative(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_positive_for_positive_x(self):
        """Test Bi'(x) > 0 for positive x."""
        x = torch.linspace(0.0, 5.0, 50)
        result = airy_bi_derivative(x)
        assert torch.all(result > 0), "Bi'(x) should be positive for x >= 0"

    def test_airy_equation(self):
        """Test that d/dx Bi'(x) = x * Bi(x) (from Airy equation y'' = xy)."""
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        h = 1e-6

        # d/dx Bi'(x) via finite differences
        bip_plus = airy_bi_derivative(x + h)
        bip_minus = airy_bi_derivative(x - h)
        second_deriv = (bip_plus - bip_minus) / (2 * h)

        # Should equal x * Bi(x)
        expected = x * airy_bi(x)

        torch.testing.assert_close(second_deriv, expected, atol=1e-4, rtol=1e-4)
