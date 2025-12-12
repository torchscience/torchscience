import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import gamma, sin_pi


class TestGamma(UnaryOperatorTestCase):
    func = staticmethod(gamma)
    op_name = "_gamma"

    # Gamma is neither odd nor even
    symmetry = None
    period = None

    # Gamma is positive for positive real inputs
    # (but can be negative between negative integers)
    bounds = None
    lower_bound = None

    known_values = {
        1.0: 1.0,  # gamma(1) = 0! = 1
        2.0: 1.0,  # gamma(2) = 1! = 1
        3.0: 2.0,  # gamma(3) = 2! = 2
        4.0: 6.0,  # gamma(4) = 3! = 6
        5.0: 24.0,  # gamma(5) = 4! = 24
        0.5: math.sqrt(math.pi),  # gamma(1/2) = sqrt(pi)
    }

    # Gamma has no zeros on the real line
    zeros = None

    # Reference: use torch.special.gamma (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.gamma(x))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    # Recurrence relation: gamma(x+1) = x * gamma(x)
    # Reflection formula: gamma(x) * gamma(1-x) = pi / sin(pi*x)
    identities = [
        # Recurrence: gamma(x+1) / gamma(x) = x (for x > 0)
        (lambda x: gamma(x + 1) / gamma(x), None),  # equals x, tested separately
    ]

    # Avoid non-positive integers where gamma has poles
    input_range = (0.1, 10.0)

    # Use values away from singularities for gradient checking
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]

    # Gamma doesn't preserve negative zero in a meaningful way
    preserves_negative_zero = False

    # For extreme values, gamma grows very fast (factorial growth)
    # so we use more modest extreme values
    extreme_values = [1e-10, 0.001, 0.1, 10.0, 20.0]

    def test_recurrence_relation(self):
        """Test gamma(x+1) = x * gamma(x)."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        lhs = gamma(x + 1)
        rhs = x * gamma(x)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_reflection_formula(self):
        """Test gamma(x) * gamma(1-x) = pi / sin(pi*x)."""
        # Avoid integers where sin(pi*x) = 0
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lhs = gamma(x) * gamma(1 - x)
        rhs = math.pi / sin_pi(x)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)

    def test_half_integers(self):
        """Test gamma at half-integer values using double factorial formula."""
        # gamma(n + 1/2) = (2n-1)!! / 2^n * sqrt(pi)
        # gamma(1/2) = sqrt(pi)
        # gamma(3/2) = 1/2 * sqrt(pi)
        # gamma(5/2) = 3/4 * sqrt(pi)
        # gamma(7/2) = 15/8 * sqrt(pi)
        sqrt_pi = math.sqrt(math.pi)
        expected = torch.tensor([
            sqrt_pi,           # gamma(0.5)
            0.5 * sqrt_pi,     # gamma(1.5)
            0.75 * sqrt_pi,    # gamma(2.5)
            1.875 * sqrt_pi,   # gamma(3.5)
        ])
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])
        torch.testing.assert_close(gamma(x), expected, atol=1e-5, rtol=1e-5)

    def test_factorial_relation(self):
        """Test gamma(n) = (n-1)! for positive integers."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        factorials = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0])
        torch.testing.assert_close(gamma(n), factorials, atol=1e-4, rtol=1e-5)

    def test_positive_for_positive_input(self):
        """Test gamma(x) > 0 for x > 0."""
        x = torch.linspace(0.01, 10.0, 100)
        output = gamma(x)
        assert torch.all(output > 0), "Gamma should be positive for positive inputs"

    def test_duplication_formula(self):
        """Test Legendre duplication formula: gamma(x) * gamma(x + 1/2) = sqrt(pi) / 2^(2x-1) * gamma(2x)."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        lhs = gamma(x) * gamma(x + 0.5)
        rhs = math.sqrt(math.pi) / (2 ** (2 * x - 1)) * gamma(2 * x)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)
