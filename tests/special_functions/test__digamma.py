import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import digamma, log_gamma


class TestDigamma(UnaryOperatorTestCase):
    func = staticmethod(digamma)
    op_name = "_digamma"

    # Digamma is neither odd nor even
    symmetry = None
    period = None
    bounds = None

    # Euler-Mascheroni constant
    EULER_MASCHERONI = 0.5772156649015329

    known_values = {
        1.0: -EULER_MASCHERONI,  # psi(1) = -gamma
        2.0: 1.0 - EULER_MASCHERONI,  # psi(2) = 1 - gamma
        3.0: 1.5 - EULER_MASCHERONI,  # psi(3) = 1 + 1/2 - gamma
        4.0: 11 / 6 - EULER_MASCHERONI,  # psi(4) = 1 + 1/2 + 1/3 - gamma
    }

    zeros = None  # digamma has no simple zeros

    # Reference: use torch.special.digamma (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.digamma(x))

    reference_atol = 1e-5
    reference_rtol = 1e-5

    # Avoid non-positive integers where digamma has poles
    input_range = (0.1, 10.0)

    # Use values away from singularities for gradient checking
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]

    preserves_negative_zero = False

    extreme_values = [1e-10, 0.001, 0.1, 10.0, 100.0]

    def test_recurrence_relation(self):
        """Test digamma(x+1) = digamma(x) + 1/x."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0])
        lhs = digamma(x + 1)
        rhs = digamma(x) + 1 / x
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_reflection_formula(self):
        """Test digamma(1-x) - digamma(x) = pi * cot(pi*x)."""
        # Avoid integers where cot has poles
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lhs = digamma(1 - x) - digamma(x)
        rhs = math.pi / torch.tan(math.pi * x)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)

    def test_derivative_of_log_gamma(self):
        """Test that digamma is the derivative of log_gamma."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True)
        y = log_gamma(x)
        y.sum().backward()
        numerical_derivative = x.grad
        analytical = digamma(x.detach())
        torch.testing.assert_close(
            numerical_derivative, analytical, atol=1e-5, rtol=1e-5
        )

    def test_half_integer_values(self):
        """Test digamma at half-integer values."""
        # psi(1/2) = -gamma - 2*ln(2)
        expected_half = -self.EULER_MASCHERONI - 2 * math.log(2)
        result = digamma(torch.tensor([0.5]))
        torch.testing.assert_close(
            result, torch.tensor([expected_half]), atol=1e-5, rtol=1e-5
        )

    def test_asymptotic_behavior(self):
        """Test asymptotic expansion: digamma(x) ~ ln(x) - 1/(2x) for large x."""
        x = torch.tensor([50.0, 100.0, 200.0])
        asymptotic = torch.log(x) - 1 / (2 * x)
        actual = digamma(x)
        # Should be close for large x
        relative_error = torch.abs(actual - asymptotic) / torch.abs(actual)
        assert torch.all(relative_error < 0.001), "Asymptotic approximation should be accurate for large x"

    def test_harmonic_number_relation(self):
        """Test psi(n) = H_{n-1} - gamma for positive integers."""
        # H_n = 1 + 1/2 + 1/3 + ... + 1/n (harmonic numbers)
        # psi(n+1) = H_n - gamma
        harmonic = torch.tensor([
            0.0,  # H_0 = 0
            1.0,  # H_1 = 1
            1.5,  # H_2 = 1 + 1/2
            11 / 6,  # H_3 = 1 + 1/2 + 1/3
            25 / 12,  # H_4
        ])
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = harmonic - self.EULER_MASCHERONI
        torch.testing.assert_close(digamma(n), expected, atol=1e-5, rtol=1e-5)

    def test_duplication_formula(self):
        """Test digamma duplication: psi(2x) = 0.5*psi(x) + 0.5*psi(x+0.5) + ln(2)."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        lhs = digamma(2 * x)
        rhs = 0.5 * digamma(x) + 0.5 * digamma(x + 0.5) + math.log(2)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)
