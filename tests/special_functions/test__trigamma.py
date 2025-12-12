import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import digamma, trigamma


class TestTrigamma(UnaryOperatorTestCase):
    func = staticmethod(trigamma)
    op_name = "_trigamma"

    # Trigamma is neither odd nor even
    symmetry = None
    period = None

    # Trigamma is always positive for positive real inputs
    lower_bound = 0.0

    # pi^2 / 6 (Basel problem)
    PI_SQUARED_OVER_6 = math.pi**2 / 6

    known_values = {
        1.0: PI_SQUARED_OVER_6,  # psi_1(1) = pi^2/6
        2.0: PI_SQUARED_OVER_6 - 1,  # psi_1(2) = pi^2/6 - 1
        3.0: PI_SQUARED_OVER_6 - 1.25,  # psi_1(3) = pi^2/6 - 1 - 1/4
        4.0: PI_SQUARED_OVER_6 - 1.25 - 1 / 9,  # psi_1(4) = pi^2/6 - 1 - 1/4 - 1/9
    }

    zeros = None  # trigamma has no zeros for positive reals

    # Reference: use torch.special.polygamma(1, x) (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.polygamma(1, x))

    reference_atol = 1e-5
    reference_rtol = 1e-5

    # Avoid non-positive integers where trigamma has poles
    input_range = (0.1, 10.0)

    # Use values away from singularities for gradient checking
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]

    preserves_negative_zero = False

    extreme_values = [1e-10, 0.001, 0.1, 10.0, 100.0]

    def test_recurrence_relation(self):
        """Test trigamma(x+1) = trigamma(x) - 1/x^2."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0])
        lhs = trigamma(x + 1)
        rhs = trigamma(x) - 1 / (x**2)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_reflection_formula(self):
        """Test trigamma(1-x) + trigamma(x) = pi^2 / sin^2(pi*x)."""
        # Avoid integers where sin has zeros
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lhs = trigamma(1 - x) + trigamma(x)
        rhs = (math.pi**2) / (torch.sin(math.pi * x) ** 2)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)

    def test_derivative_of_digamma(self):
        """Test that trigamma is the derivative of digamma."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True)
        y = digamma(x)
        y.sum().backward()
        numerical_derivative = x.grad
        analytical = trigamma(x.detach())
        torch.testing.assert_close(
            numerical_derivative, analytical, atol=1e-5, rtol=1e-5
        )

    def test_half_integer_values(self):
        """Test trigamma at half-integer values."""
        # psi_1(1/2) = pi^2/2
        expected_half = math.pi**2 / 2
        result = trigamma(torch.tensor([0.5]))
        torch.testing.assert_close(
            result, torch.tensor([expected_half]), atol=1e-5, rtol=1e-5
        )

    def test_series_representation(self):
        """Test series representation: trigamma(x) = sum_{k=0}^{inf} 1/(x+k)^2."""
        x = torch.tensor([1.0, 2.0, 3.0])
        # Compute partial sum (enough terms for convergence)
        partial_sum = torch.zeros_like(x)
        for k in range(1000):
            partial_sum += 1 / (x + k) ** 2
        torch.testing.assert_close(trigamma(x), partial_sum, atol=1e-3, rtol=1e-3)

    def test_asymptotic_behavior(self):
        """Test asymptotic expansion: trigamma(x) ~ 1/x + 1/(2x^2) for large x."""
        x = torch.tensor([50.0, 100.0, 200.0])
        asymptotic = 1 / x + 1 / (2 * x**2) + 1 / (6 * x**3)
        actual = trigamma(x)
        # Should be close for large x
        relative_error = torch.abs(actual - asymptotic) / torch.abs(actual)
        assert torch.all(relative_error < 0.001), "Asymptotic approximation should be accurate for large x"

    def test_positivity(self):
        """Test trigamma(x) > 0 for all x > 0."""
        x = torch.linspace(0.01, 10.0, 100)
        output = trigamma(x)
        assert torch.all(output > 0), "Trigamma should be positive for positive inputs"

    def test_monotonically_decreasing(self):
        """Test trigamma is monotonically decreasing for x > 0."""
        x = torch.linspace(0.1, 10.0, 100)
        output = trigamma(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff < 0), "Trigamma should be monotonically decreasing for x > 0"

    def test_duplication_formula(self):
        """Test trigamma duplication: psi_1(2x) = 0.25*(psi_1(x) + psi_1(x+0.5))."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        lhs = trigamma(2 * x)
        rhs = 0.25 * (trigamma(x) + trigamma(x + 0.5))
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)
