import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import inverse_erfc


class TestInverseErfc(UnaryOperatorTestCase):
    func = staticmethod(inverse_erfc)
    op_name = "_inverse_erfc"

    # inverse_erfc is neither odd nor even
    symmetry = None
    period = None

    # inverse_erfc is unbounded
    bounds = None

    known_values = {
        1.0: 0.0,  # inverse_erfc(1) = 0
        0.5: 0.4769362762044699,  # inverse_erfc(0.5)
        1.5: -0.4769362762044699,  # inverse_erfc(1.5)
    }

    zeros = [1.0]  # inverse_erfc(1) = 0

    # Reference: use scipy.special.erfcinv
    reference = None  # PyTorch doesn't have built-in erfcinv

    reference_atol = 1e-6
    reference_rtol = 1e-5

    # Valid input range is (0, 2) exclusive
    input_range = (0.01, 1.99)

    gradcheck_inputs = [0.1, 0.5, 1.0, 1.5, 1.9]

    preserves_negative_zero = False

    extreme_values = [0.001, 0.01, 0.1, 1.9, 1.99, 1.999]

    def test_inverse_property(self):
        """Test that erfc(inverse_erfc(x)) = x."""
        x = torch.tensor([0.1, 0.5, 1.0, 1.5, 1.9])
        y = inverse_erfc(x)
        reconstructed = torch.special.erfc(y)
        torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)

    def test_inverse_property_reverse(self):
        """Test that inverse_erfc(erfc(x)) = x."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = torch.special.erfc(x)
        reconstructed = inverse_erfc(y)
        torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)

    def test_symmetry_around_one(self):
        """Test inverse_erfc(1 + x) = -inverse_erfc(1 - x)."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        torch.testing.assert_close(
            inverse_erfc(1 + x), -inverse_erfc(1 - x), atol=1e-6, rtol=1e-6
        )

    def test_at_one(self):
        """Test inverse_erfc(1) = 0."""
        result = inverse_erfc(torch.tensor([1.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-10, rtol=0)

    def test_limits(self):
        """Test inverse_erfc approaches ±inf at boundaries."""
        # inverse_erfc(x) -> inf as x -> 0+
        small_pos = torch.tensor([0.001, 0.0001])
        result_small = inverse_erfc(small_pos)
        assert torch.all(result_small > 2.0), "inverse_erfc should be large for small x"

        # inverse_erfc(x) -> -inf as x -> 2-
        near_two = torch.tensor([1.999, 1.9999])
        result_near_two = inverse_erfc(near_two)
        assert torch.all(
            result_near_two < -2.0
        ), "inverse_erfc should be large negative for x near 2"

    def test_monotonically_decreasing(self):
        """Test inverse_erfc is monotonically decreasing on (0, 2)."""
        x = torch.linspace(0.01, 1.99, 100)
        output = inverse_erfc(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff <= 0), "inverse_erfc should be monotonically decreasing"

    def test_derivative(self):
        """Test d/dx inverse_erfc(x) = -sqrt(pi)/2 * exp(inverse_erfc(x)^2)."""
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
        y = inverse_erfc(x)
        y.sum().backward()

        sqrt_pi_over_2 = math.sqrt(math.pi) / 2
        y_detached = inverse_erfc(x.detach())
        expected_grad = -sqrt_pi_over_2 * torch.exp(y_detached**2)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_special_values_float64(self):
        """Test special values with float64 precision."""
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        result = inverse_erfc(x)
        expected = torch.tensor(
            [0.4769362762044699, 0.0, -0.4769362762044699], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_relation_to_inverse_erf(self):
        """Test inverse_erfc(x) = inverse_erf(1 - x)."""
        x = torch.tensor([0.5, 1.0, 1.5])
        # inverse_erf is not in torch.special, so we verify via round-trip:
        # erf(inverse_erfc(x)) should equal 1 - x
        y = inverse_erfc(x)
        erf_y = torch.special.erf(y)
        expected = 1 - x
        torch.testing.assert_close(erf_y, expected, atol=1e-6, rtol=1e-6)

    def test_normal_quantile_relation(self):
        """Test relation to normal distribution quantile function.

        For standard normal: quantile(p) = sqrt(2) * inverse_erfc(2 * (1 - p))
        """
        p = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        # Normal quantile via inverse_erfc
        quantile = math.sqrt(2) * inverse_erfc(2 * (1 - p))
        # Compare with PyTorch's implementation
        expected = torch.distributions.Normal(0, 1).icdf(p)
        torch.testing.assert_close(quantile, expected, atol=1e-5, rtol=1e-5)
