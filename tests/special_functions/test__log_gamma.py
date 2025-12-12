import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import gamma, log_gamma


class TestLogGamma(UnaryOperatorTestCase):
    func = staticmethod(log_gamma)
    op_name = "_log_gamma"

    # log_gamma is neither odd nor even
    symmetry = None
    period = None
    bounds = None

    known_values = {
        1.0: 0.0,  # log(gamma(1)) = log(1) = 0
        2.0: 0.0,  # log(gamma(2)) = log(1) = 0
        3.0: math.log(2.0),  # log(gamma(3)) = log(2)
        4.0: math.log(6.0),  # log(gamma(4)) = log(6)
        5.0: math.log(24.0),  # log(gamma(5)) = log(24)
        0.5: math.log(math.sqrt(math.pi)),  # log(gamma(1/2)) = log(sqrt(pi))
    }

    zeros = [1.0, 2.0]  # log_gamma(1) = log_gamma(2) = 0

    # Reference: use torch.special.gammaln (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.gammaln(x))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    # Avoid non-positive integers where gamma has poles
    input_range = (0.1, 20.0)

    # Use values away from singularities for gradient checking
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]

    preserves_negative_zero = False

    # log_gamma is more stable for large values than gamma
    extreme_values = [1e-10, 0.001, 0.1, 50.0, 100.0, 170.0]

    def test_relation_to_gamma(self):
        """Test log_gamma(x) = log(gamma(x)) for positive gamma values."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        # For these values, gamma(x) > 0, so log(gamma(x)) is well-defined
        torch.testing.assert_close(
            log_gamma(x), torch.log(gamma(x)), atol=1e-5, rtol=1e-5
        )

    def test_numerical_stability_large_values(self):
        """Test log_gamma is stable for large values where gamma would overflow."""
        # gamma(171) overflows float64, but log_gamma should handle it
        x = torch.tensor([100.0, 150.0, 170.0, 171.0, 200.0])
        output = log_gamma(x)
        assert torch.all(torch.isfinite(output)), "log_gamma should be stable for large inputs"
        # Verify monotonically increasing for large positive values
        assert torch.all(output[1:] > output[:-1]), "log_gamma should increase for large x"

    def test_reflection_formula(self):
        """Test log_gamma reflection: log|gamma(x)| + log|gamma(1-x)| = log(pi/|sin(pi*x)|)."""
        # For x in (0, 1), both gamma(x) and gamma(1-x) are positive
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lhs = log_gamma(x) + log_gamma(1 - x)
        rhs = torch.log(torch.tensor(math.pi)) - torch.log(torch.abs(torch.sin(math.pi * x)))
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)

    def test_recurrence_relation(self):
        """Test log_gamma(x+1) = log_gamma(x) + log(x)."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0])
        lhs = log_gamma(x + 1)
        rhs = log_gamma(x) + torch.log(x)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_stirling_approximation(self):
        """Test Stirling's approximation for large x: log_gamma(x) ~ (x-0.5)*log(x) - x + 0.5*log(2*pi)."""
        x = torch.tensor([50.0, 100.0, 200.0])
        stirling = (x - 0.5) * torch.log(x) - x + 0.5 * math.log(2 * math.pi)
        actual = log_gamma(x)
        # Stirling is asymptotically accurate
        relative_error = torch.abs(actual - stirling) / torch.abs(actual)
        assert torch.all(relative_error < 0.001), "Stirling approximation should be accurate for large x"

    def test_factorial_logarithms(self):
        """Test log_gamma(n) = log((n-1)!) for positive integers."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        log_factorials = torch.tensor([
            0.0,  # log(0!)
            0.0,  # log(1!)
            math.log(2),  # log(2!)
            math.log(6),  # log(3!)
            math.log(24),  # log(4!)
            math.log(120),  # log(5!)
            math.log(720),  # log(6!)
            math.log(5040),  # log(7!)
        ])
        torch.testing.assert_close(log_gamma(n), log_factorials, atol=1e-5, rtol=1e-5)
