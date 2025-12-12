import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import airy_ai, airy_bi


class TestAiryBi(UnaryOperatorTestCase):
    func = staticmethod(airy_bi)
    op_name = "_airy_bi"

    # airy_bi is neither odd nor even
    symmetry = None
    period = None

    # airy_bi is unbounded (grows exponentially for x > 0, oscillates for x < 0)
    bounds = None

    # Bi(0) = 1 / (3^(1/6) * Gamma(2/3)) ≈ 0.6149266274
    known_values = {
        0.0: 0.6149266274460007,
    }

    zeros = []  # Zeros are at specific negative values

    # Reference: use scipy.special.airy if available
    reference = None

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 5.0)  # Limited positive range due to exponential growth

    gradcheck_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]

    preserves_negative_zero = True

    extreme_values = [-10.0, -5.0, 0.0, 2.0, 3.0]

    def test_at_zero(self):
        """Test Bi(0) = 1 / (3^(1/6) * Gamma(2/3))."""
        result = airy_bi(torch.tensor([0.0], dtype=torch.float64))
        # Bi(0) ≈ 0.6149266274460007
        expected = torch.tensor([0.6149266274460007], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_exponential_growth_positive_x(self):
        """Test Bi(x) grows exponentially for large positive x."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = airy_bi(x)
        # All values should be positive and increasing
        assert torch.all(result > 0), "Bi(x) should be positive for x > 0"
        assert torch.all(
            result[1:] > result[:-1]
        ), "Bi(x) should increase for increasing positive x"

    def test_oscillation_negative_x(self):
        """Test Bi(x) oscillates for negative x."""
        # Bi has zeros at approximately -1.174, -3.271, -4.831, ...
        x = torch.linspace(-10.0, -1.0, 100)
        result = airy_bi(x)
        # Check that there are sign changes (oscillation)
        sign_changes = torch.sum(result[1:] * result[:-1] < 0)
        assert sign_changes > 0, "Bi(x) should oscillate for x < 0"

    def test_first_zero(self):
        """Test Bi(x) ≈ 0 at first zero (x ≈ -1.174)."""
        # First zero of Bi is at approximately -1.1737274176
        x = torch.tensor([-1.1737274176], dtype=torch.float64)
        result = airy_bi(x)
        torch.testing.assert_close(
            result, torch.tensor([0.0], dtype=torch.float64), atol=1e-6, rtol=0
        )

    def test_derivative_is_airy_bi_prime(self):
        """Test d/dx Bi(x) = Bi'(x)."""
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
        y = airy_bi(x)
        y.sum().backward()

        # Bi'(0) = 3^(1/6) / Gamma(1/3) ≈ 0.4482883573
        # We just check the gradient exists and is finite
        assert torch.all(torch.isfinite(x.grad)), "Gradient should be finite"

    def test_asymptotic_positive(self):
        """Test asymptotic behavior for large positive x.

        Bi(x) ~ exp(2/3 * x^(3/2)) / (sqrt(pi) * x^(1/4)) as x -> +inf
        """
        x = torch.tensor([5.0], dtype=torch.float64)
        result = airy_bi(x)
        # Asymptotic approximation
        asymptotic = torch.exp(2.0 / 3.0 * x ** (3.0 / 2.0)) / (
            math.sqrt(math.pi) * x ** (1.0 / 4.0)
        )
        # Should be reasonably close for x = 5
        ratio = result / asymptotic
        assert 0.9 < ratio.item() < 1.1, "Bi(x) should match asymptotic for large x"

    def test_differential_equation(self):
        """Test Bi satisfies y'' - xy = 0 approximately via finite differences."""
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        h = 1e-5

        # Compute second derivative via finite differences
        bi_x = airy_bi(x)
        bi_x_plus = airy_bi(x + h)
        bi_x_minus = airy_bi(x - h)
        bi_second_deriv = (bi_x_plus - 2 * bi_x + bi_x_minus) / (h * h)

        # y'' should equal x * y
        expected = x * bi_x
        torch.testing.assert_close(bi_second_deriv, expected, atol=1e-4, rtol=1e-4)

    def test_specific_values(self):
        """Test Bi at specific known values."""
        test_cases = [
            (0.0, 0.6149266274460007),
            (1.0, 1.2074235949528713),
            (2.0, 3.2980949999782147),
            (-1.0, 0.10399738949694461),
            (-2.0, -0.41230258795639087),
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = airy_bi(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_at_zero(self):
        """Test Bi'(0) = 3^(1/6) / Gamma(1/3)."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = airy_bi(x)
        y.backward()
        # Bi'(0) ≈ 0.4482883573538264
        expected_grad = torch.tensor([0.4482883573538264], dtype=torch.float64)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-6)

    def test_positive_for_positive_x(self):
        """Test Bi(x) > 0 for all x >= 0."""
        x = torch.linspace(0.0, 10.0, 100)
        result = airy_bi(x)
        assert torch.all(result > 0), "Bi(x) should be positive for x >= 0"

    def test_wronskian(self):
        """Test the Wronskian W(Ai, Bi) = Ai(x)*Bi'(x) - Ai'(x)*Bi(x) = 1/pi."""
        x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)

        # Compute Ai(x) and its derivative
        ai_x = airy_ai(x)
        ai_x.sum().backward()
        ai_prime = x.grad.clone()

        # Reset gradient
        x.grad.zero_()

        # Compute Bi(x) and its derivative
        bi_x = airy_bi(x)
        bi_x.sum().backward()
        bi_prime = x.grad.clone()

        # Wronskian should be 1/pi
        wronskian = ai_x * bi_prime - ai_prime * bi_x
        expected = torch.full_like(wronskian, 1.0 / math.pi)
        torch.testing.assert_close(wronskian, expected, atol=1e-5, rtol=1e-5)

    def test_relation_to_airy_ai(self):
        """Test that Ai and Bi are linearly independent solutions."""
        x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        ai_vals = airy_ai(x)
        bi_vals = airy_bi(x)
        # They should not be proportional (different ratios at different points)
        ratios = bi_vals / ai_vals
        # Ratios should vary (not constant)
        assert not torch.allclose(
            ratios, ratios[0] * torch.ones_like(ratios)
        ), "Ai and Bi should be linearly independent"
