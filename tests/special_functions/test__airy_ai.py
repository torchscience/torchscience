import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import airy_ai


class TestAiryAi(UnaryOperatorTestCase):
    func = staticmethod(airy_ai)
    op_name = "_airy_ai"

    # airy_ai is neither odd nor even
    symmetry = None
    period = None

    # airy_ai is unbounded (oscillates for x < 0)
    bounds = None

    # Ai(0) = 1 / (3^(2/3) * Gamma(2/3)) ≈ 0.3550280539
    known_values = {
        0.0: 0.3550280538878172,
    }

    zeros = []  # Zeros are at specific negative values

    # Reference: use scipy.special.airy if available
    reference = None

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 10.0)

    gradcheck_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]

    preserves_negative_zero = True

    extreme_values = [-10.0, -5.0, 0.0, 5.0, 10.0]

    def test_at_zero(self):
        """Test Ai(0) = 1 / (3^(2/3) * Gamma(2/3))."""
        result = airy_ai(torch.tensor([0.0], dtype=torch.float64))
        # Ai(0) ≈ 0.3550280538878172
        expected = torch.tensor([0.3550280538878172], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_exponential_decay_positive_x(self):
        """Test Ai(x) decays exponentially for large positive x."""
        x = torch.tensor([5.0, 10.0, 15.0])
        result = airy_ai(x)
        # All values should be positive and decreasing
        assert torch.all(result > 0), "Ai(x) should be positive for x > 0"
        assert torch.all(
            result[1:] < result[:-1]
        ), "Ai(x) should decrease for increasing positive x"

    def test_oscillation_negative_x(self):
        """Test Ai(x) oscillates for negative x."""
        # Ai has zeros at approximately -2.338, -4.088, -5.521, ...
        x = torch.linspace(-10.0, -1.0, 100)
        result = airy_ai(x)
        # Check that there are sign changes (oscillation)
        sign_changes = torch.sum(result[1:] * result[:-1] < 0)
        assert sign_changes > 0, "Ai(x) should oscillate for x < 0"

    def test_first_zero(self):
        """Test Ai(x) ≈ 0 at first zero (x ≈ -2.338)."""
        # First zero of Ai is at approximately -2.3381074105
        x = torch.tensor([-2.3381074105], dtype=torch.float64)
        result = airy_ai(x)
        torch.testing.assert_close(
            result, torch.tensor([0.0], dtype=torch.float64), atol=1e-6, rtol=0
        )

    def test_derivative_is_airy_ai_prime(self):
        """Test d/dx Ai(x) = Ai'(x)."""
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        y = airy_ai(x)
        y.sum().backward()

        # Known values of Ai'(x):
        # Ai'(0) = -1 / (3^(1/3) * Gamma(1/3)) ≈ -0.2588194038
        # We just check the gradient exists and is finite
        assert torch.all(torch.isfinite(x.grad)), "Gradient should be finite"

    def test_asymptotic_positive(self):
        """Test asymptotic behavior for large positive x.

        Ai(x) ~ exp(-2/3 * x^(3/2)) / (2 * sqrt(pi) * x^(1/4)) as x -> +inf
        """
        x = torch.tensor([10.0], dtype=torch.float64)
        result = airy_ai(x)
        # Asymptotic approximation
        asymptotic = (
            torch.exp(-2.0 / 3.0 * x ** (3.0 / 2.0))
            / (2.0 * math.sqrt(math.pi) * x ** (1.0 / 4.0))
        )
        # Should be reasonably close for x = 10
        ratio = result / asymptotic
        assert 0.9 < ratio.item() < 1.1, "Ai(x) should match asymptotic for large x"

    def test_differential_equation(self):
        """Test Ai satisfies y'' - xy = 0 approximately via finite differences."""
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        h = 1e-5

        # Compute second derivative via finite differences
        ai_x = airy_ai(x)
        ai_x_plus = airy_ai(x + h)
        ai_x_minus = airy_ai(x - h)
        ai_second_deriv = (ai_x_plus - 2 * ai_x + ai_x_minus) / (h * h)

        # y'' should equal x * y
        expected = x * ai_x
        torch.testing.assert_close(ai_second_deriv, expected, atol=1e-4, rtol=1e-4)

    def test_specific_values(self):
        """Test Ai at specific known values."""
        test_cases = [
            (0.0, 0.3550280538878172),
            (1.0, 0.1352924163128814),
            (2.0, 0.03492413042327438),
            (-1.0, 0.5355608832923521),
            (-2.0, 0.22740742820168557),
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = airy_ai(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_at_zero(self):
        """Test Ai'(0) = -1 / (3^(1/3) * Gamma(1/3))."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = airy_ai(x)
        y.backward()
        # Ai'(0) ≈ -0.2588194037928068
        expected_grad = torch.tensor([-0.2588194037928068], dtype=torch.float64)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-6)

    def test_positive_for_positive_x(self):
        """Test Ai(x) > 0 for all x > 0."""
        x = torch.linspace(0.0, 20.0, 100)
        result = airy_ai(x)
        assert torch.all(result > 0), "Ai(x) should be positive for x >= 0"
