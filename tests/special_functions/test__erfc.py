import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import erfc


class TestErfc(UnaryOperatorTestCase):
    func = staticmethod(erfc)
    op_name = "_erfc"

    # erfc is neither odd nor even
    symmetry = None
    period = None

    # erfc is bounded between 0 and 2
    bounds = (0.0, 2.0)

    known_values = {
        0.0: 1.0,
        1.0: 0.1572992070502851,  # erfc(1) = 1 - erf(1)
        2.0: 0.004677734981047266,  # erfc(2)
        3.0: 2.2090496764229446e-05,  # erfc(3)
    }

    zeros = []  # erfc has no zeros for finite x

    # Reference: use torch.special.erfc (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.erfc(x))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-5.0, 5.0)

    gradcheck_inputs = [0.1, 0.5, 1.0, 2.0]

    preserves_negative_zero = False

    extreme_values = [1e-30, 1e-10, 1e-5, 5.0, 10.0]

    def test_relation_to_erf(self):
        """Test erfc(x) = 1 - erf(x)."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
        torch.testing.assert_close(
            erfc(x), 1 - torch.special.erf(x), atol=1e-7, rtol=1e-6
        )

    def test_reflection_property(self):
        """Test erfc(x) + erfc(-x) = 2."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
        torch.testing.assert_close(
            erfc(x) + erfc(-x), 2 * torch.ones_like(x), atol=1e-7, rtol=1e-6
        )

    def test_limits(self):
        """Test erfc approaches 0 and 2 for large |x|."""
        large_pos = torch.tensor([5.0, 10.0, 20.0])
        large_neg = torch.tensor([-5.0, -10.0, -20.0])

        # erfc(x) -> 0 as x -> inf
        torch.testing.assert_close(
            erfc(large_pos), torch.zeros_like(large_pos), atol=1e-6, rtol=1e-6
        )

        # erfc(x) -> 2 as x -> -inf
        torch.testing.assert_close(
            erfc(large_neg), 2 * torch.ones_like(large_neg), atol=1e-6, rtol=1e-6
        )

    def test_derivative(self):
        """Test d/dx erfc(x) = -(2/sqrt(pi)) * exp(-x^2)."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
        y = erfc(x)
        y.sum().backward()

        two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
        expected_grad = -two_over_sqrt_pi * torch.exp(-x.detach() ** 2)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_monotonically_decreasing(self):
        """Test erfc is monotonically decreasing."""
        x = torch.linspace(-5.0, 5.0, 100)
        output = erfc(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff <= 0), "erfc should be monotonically decreasing"

    def test_special_values(self):
        """Test erfc at special values."""
        # erfc(0) = 1
        torch.testing.assert_close(
            erfc(torch.tensor([0.0])), torch.tensor([1.0]), atol=1e-10, rtol=0
        )

        # erfc(inf) = 0
        torch.testing.assert_close(
            erfc(torch.tensor([float("inf")])),
            torch.tensor([0.0]),
            atol=1e-10,
            rtol=0,
        )

        # erfc(-inf) = 2
        torch.testing.assert_close(
            erfc(torch.tensor([float("-inf")])),
            torch.tensor([2.0]),
            atol=1e-10,
            rtol=0,
        )

    def test_numerical_stability_large_positive(self):
        """Test numerical stability for large positive x where erfc is tiny."""
        # For large x, erfc(x) is very small but should not be exactly 0
        x = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        result = erfc(x)
        # All values should be positive (not zero due to underflow)
        assert torch.all(result > 0), "erfc should not underflow to exactly zero"
        # Check against known values
        expected = torch.tensor(
            [2.2090496764229446e-05, 1.541725790028002e-08, 1.5374597944280347e-12],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, atol=1e-16, rtol=1e-4)

    def test_at_zero(self):
        """Test erfc(0) = 1 exactly."""
        assert erfc(torch.tensor([0.0])).item() == 1.0

    def test_relation_to_normal_cdf(self):
        """Test relation to standard normal survival function: 1 - Phi(x) = 0.5 * erfc(x/sqrt(2))."""
        x = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
        # Standard normal survival function (1 - CDF)
        survival = 0.5 * erfc(x / math.sqrt(2))
        # Compare with torch's implementation
        expected = 1 - torch.distributions.Normal(0, 1).cdf(x)
        torch.testing.assert_close(survival, expected, atol=1e-5, rtol=1e-5)
