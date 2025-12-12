import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import sinhc_pi


class TestSinhcPi(UnaryOperatorTestCase):
    func = staticmethod(sinhc_pi)
    op_name = "_sinhc_pi"

    # sinhc_pi is an even function
    symmetry = "even"
    period = None

    # sinhc_pi is unbounded (grows exponentially)
    bounds = None

    known_values = {
        0.0: 1.0,  # sinhc_pi(0) = 1
    }

    zeros = []  # sinhc_pi has no zeros

    # Reference: compute manually as sinh(pi*x)/(pi*x)
    reference = None

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-3.0, 3.0)

    gradcheck_inputs = [0.1, 0.25, 0.5, 1.0, 1.5]

    preserves_negative_zero = True

    extreme_values = [1e-10, 1e-5, 0.001, 2.0, 3.0]

    def test_at_zero(self):
        """Test sinhc_pi(0) = 1."""
        result = sinhc_pi(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([1.0]), atol=1e-10, rtol=0)

    def test_symmetry_even(self):
        """Test sinhc_pi(-x) = sinhc_pi(x) (even function)."""
        x = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0])
        torch.testing.assert_close(sinhc_pi(-x), sinhc_pi(x), atol=1e-7, rtol=1e-6)

    def test_always_positive(self):
        """Test sinhc_pi(x) > 0 for all real x."""
        x = torch.linspace(-5.0, 5.0, 100)
        result = sinhc_pi(x)
        assert torch.all(result > 0), "sinhc_pi should always be positive"

    def test_minimum_at_zero(self):
        """Test sinhc_pi(x) >= 1 for all real x."""
        x = torch.linspace(-5.0, 5.0, 100)
        result = sinhc_pi(x)
        assert torch.all(result >= 1.0 - 1e-6), "sinhc_pi should have minimum value 1"

    def test_derivative_at_zero(self):
        """Test d/dx sinhc_pi(x)|_{x=0} = 0."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = sinhc_pi(x)
        y.backward()
        torch.testing.assert_close(
            x.grad, torch.tensor([0.0], dtype=torch.float64), atol=1e-10, rtol=0
        )

    def test_derivative(self):
        """Test d/dx sinhc_pi(x) = (cosh(pi*x) - sinhc_pi(x)) / x."""
        x = torch.tensor([0.25, 0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
        y = sinhc_pi(x)
        y.sum().backward()

        x_detached = x.detach()
        pi_x = math.pi * x_detached
        expected_grad = (torch.cosh(pi_x) - sinhc_pi(x_detached)) / x_detached
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_relation_to_sinh(self):
        """Test sinhc_pi(x) = sinh(pi*x) / (pi*x) for x != 0."""
        x = torch.tensor([0.1, 0.25, 0.5, 1.0, 1.5])
        pi_x = math.pi * x
        expected = torch.sinh(pi_x) / pi_x
        torch.testing.assert_close(sinhc_pi(x), expected, atol=1e-7, rtol=1e-6)

    def test_exponential_growth(self):
        """Test sinhc_pi grows exponentially for large x."""
        x = torch.tensor([2.0, 3.0, 4.0])
        result = sinhc_pi(x)
        # sinhc_pi(x) ~ exp(pi*x) / (2*pi*x) for large x
        asymptotic = torch.exp(math.pi * x) / (2 * math.pi * x)
        # Should be within factor of 2 for these values
        ratio = result / asymptotic
        assert torch.all(ratio > 0.5) and torch.all(
            ratio < 2.0
        ), "sinhc_pi should grow exponentially"

    def test_small_x_taylor(self):
        """Test sinhc_pi(x) ≈ 1 + (pi*x)^2/6 for small x."""
        x = torch.tensor([1e-4, 1e-3, 1e-2])
        taylor_approx = 1 + (math.pi * x) ** 2 / 6
        torch.testing.assert_close(sinhc_pi(x), taylor_approx, atol=1e-6, rtol=1e-4)

    def test_numerical_stability_near_zero(self):
        """Test numerical stability for very small x."""
        small_x = torch.tensor([1e-15, 1e-10, 1e-8, 1e-6])
        result = sinhc_pi(small_x)
        # Should be very close to 1 for small x
        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-5, rtol=1e-5)

    def test_monotonically_increasing_for_positive_x(self):
        """Test sinhc_pi is monotonically increasing for x > 0."""
        x = torch.linspace(0.01, 5.0, 100)
        output = sinhc_pi(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff >= 0), "sinhc_pi should be monotonically increasing for x > 0"

    def test_specific_values(self):
        """Test sinhc_pi at specific values."""
        # sinhc_pi(1) = sinh(pi) / pi ≈ 3.6769
        x = torch.tensor([1.0])
        expected = torch.tensor([math.sinh(math.pi) / math.pi])
        torch.testing.assert_close(sinhc_pi(x), expected, atol=1e-4, rtol=1e-4)

        # sinhc_pi(0.5) = sinh(pi/2) / (pi/2) ≈ 1.5924
        x = torch.tensor([0.5])
        expected = torch.tensor([math.sinh(math.pi / 2) / (math.pi / 2)])
        torch.testing.assert_close(sinhc_pi(x), expected, atol=1e-4, rtol=1e-4)

    def test_complex_input(self):
        """Test sinhc_pi with complex input."""
        z = torch.tensor([0.5 + 0.5j, 1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
        result = sinhc_pi(z)
        # sinhc_pi should return complex values
        assert result.dtype == torch.complex64
