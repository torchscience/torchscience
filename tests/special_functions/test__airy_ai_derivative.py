import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import airy_ai, airy_ai_derivative


class TestAiryAiDerivative(UnaryOperatorTestCase):
    func = staticmethod(airy_ai_derivative)
    op_name = "_airy_ai_derivative"

    symmetry = None
    period = None
    bounds = None

    # Ai'(0) = -1 / (3^(1/3) * Gamma(1/3)) ≈ -0.2588194038
    known_values = {
        0.0: -0.2588194037928068,
    }

    zeros = []

    reference = None
    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 10.0)
    gradcheck_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    extreme_values = [-10.0, -5.0, 0.0, 5.0, 10.0]

    def test_at_zero(self):
        """Test Ai'(0) = -1 / (3^(1/3) * Gamma(1/3))."""
        result = airy_ai_derivative(torch.tensor([0.0], dtype=torch.float64))
        expected = torch.tensor([-0.2588194037928068], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_differential_equation(self):
        """Test Ai' is the derivative of Ai via finite differences."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        h = 1e-6

        # Numerical derivative of Ai
        ai_plus = airy_ai(x + h)
        ai_minus = airy_ai(x - h)
        numerical_deriv = (ai_plus - ai_minus) / (2 * h)

        # Ai'(x) from the function
        result = airy_ai_derivative(x)

        torch.testing.assert_close(result, numerical_deriv, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test Ai' at specific known values."""
        test_cases = [
            (0.0, -0.2588194037928068),
            (1.0, -0.1591474412967932),
            (2.0, -0.05309038443365353),
            (-1.0, -0.01016056711664515),
            (-2.0, 0.6182590188621264),
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = airy_ai_derivative(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_negative_for_small_positive_x(self):
        """Test Ai'(x) < 0 for small positive x."""
        x = torch.linspace(0.0, 5.0, 50)
        result = airy_ai_derivative(x)
        assert torch.all(result < 0), "Ai'(x) should be negative for small positive x"

    def test_airy_equation(self):
        """Test that d/dx Ai'(x) = x * Ai(x) (from Airy equation y'' = xy)."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        h = 1e-6

        # d/dx Ai'(x) via finite differences
        aip_plus = airy_ai_derivative(x + h)
        aip_minus = airy_ai_derivative(x - h)
        second_deriv = (aip_plus - aip_minus) / (2 * h)

        # Should equal x * Ai(x)
        expected = x * airy_ai(x)

        torch.testing.assert_close(second_deriv, expected, atol=1e-5, rtol=1e-5)
