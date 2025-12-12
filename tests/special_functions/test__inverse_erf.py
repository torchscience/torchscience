import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import inverse_erf


class TestInverseErf(UnaryOperatorTestCase):
    func = staticmethod(inverse_erf)
    op_name = "_inverse_erf"

    # inverse_erf is an odd function
    symmetry = "odd"
    period = None

    # inverse_erf is unbounded
    bounds = None

    known_values = {
        0.0: 0.0,  # inverse_erf(0) = 0
        0.5: 0.4769362762044699,  # inverse_erf(0.5)
        -0.5: -0.4769362762044699,  # inverse_erf(-0.5)
    }

    zeros = [0.0]  # inverse_erf(0) = 0

    # Reference: use torch.special.erfinv (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.erfinv(x))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    # Valid input range is (-1, 1) exclusive
    input_range = (-0.99, 0.99)

    gradcheck_inputs = [-0.5, 0.0, 0.5, 0.9]

    preserves_negative_zero = True

    extreme_values = [0.001, 0.01, 0.1, 0.9, 0.99, 0.999]

    def test_inverse_property(self):
        """Test that erf(inverse_erf(x)) = x."""
        x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 0.9])
        y = inverse_erf(x)
        reconstructed = torch.special.erf(y)
        torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)

    def test_inverse_property_reverse(self):
        """Test that inverse_erf(erf(x)) = x."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = torch.special.erf(x)
        reconstructed = inverse_erf(y)
        torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)

    def test_symmetry_odd(self):
        """Test inverse_erf(-x) = -inverse_erf(x) (odd function)."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        torch.testing.assert_close(inverse_erf(-x), -inverse_erf(x), atol=1e-7, rtol=1e-6)

    def test_at_zero(self):
        """Test inverse_erf(0) = 0."""
        result = inverse_erf(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-10, rtol=0)

    def test_limits(self):
        """Test inverse_erf approaches ±inf at boundaries."""
        # inverse_erf(x) -> inf as x -> 1-
        near_one = torch.tensor([0.99, 0.999])
        result_near_one = inverse_erf(near_one)
        assert torch.all(result_near_one > 1.5), "inverse_erf should be large for x near 1"

        # inverse_erf(x) -> -inf as x -> -1+
        near_neg_one = torch.tensor([-0.99, -0.999])
        result_near_neg_one = inverse_erf(near_neg_one)
        assert torch.all(
            result_near_neg_one < -1.5
        ), "inverse_erf should be large negative for x near -1"

    def test_monotonically_increasing(self):
        """Test inverse_erf is monotonically increasing on (-1, 1)."""
        x = torch.linspace(-0.99, 0.99, 100)
        output = inverse_erf(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff >= 0), "inverse_erf should be monotonically increasing"

    def test_derivative(self):
        """Test d/dx inverse_erf(x) = sqrt(pi)/2 * exp(inverse_erf(x)^2)."""
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64, requires_grad=True)
        y = inverse_erf(x)
        y.sum().backward()

        sqrt_pi_over_2 = math.sqrt(math.pi) / 2
        y_detached = inverse_erf(x.detach())
        expected_grad = sqrt_pi_over_2 * torch.exp(y_detached**2)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_special_values_float64(self):
        """Test special values with float64 precision."""
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = inverse_erf(x)
        expected = torch.tensor(
            [-0.4769362762044699, 0.0, 0.4769362762044699], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_relation_to_inverse_erfc(self):
        """Test inverse_erf(x) = inverse_erfc(1 - x)."""
        from torchscience.special_functions import inverse_erfc

        x = torch.tensor([-0.5, 0.0, 0.5])
        torch.testing.assert_close(
            inverse_erf(x), inverse_erfc(1 - x), atol=1e-6, rtol=1e-6
        )

    def test_normal_quantile_relation(self):
        """Test relation to normal distribution quantile function.

        For standard normal: quantile(p) = sqrt(2) * inverse_erf(2*p - 1)
        """
        p = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        # Normal quantile via inverse_erf
        quantile = math.sqrt(2) * inverse_erf(2 * p - 1)
        # Compare with PyTorch's implementation
        expected = torch.distributions.Normal(0, 1).icdf(p)
        torch.testing.assert_close(quantile, expected, atol=1e-5, rtol=1e-5)

    def test_known_erf_values(self):
        """Test inverse_erf at known erf values."""
        # erf(1) ≈ 0.8427007929497149
        x = torch.tensor([0.8427007929497149])
        result = inverse_erf(x)
        torch.testing.assert_close(result, torch.tensor([1.0]), atol=1e-5, rtol=1e-5)

        # erf(2) ≈ 0.9953222650189527
        x = torch.tensor([0.9953222650189527])
        result = inverse_erf(x)
        torch.testing.assert_close(result, torch.tensor([2.0]), atol=1e-5, rtol=1e-5)

    def test_reference_pytorch(self):
        """Test against PyTorch's built-in torch.special.erfinv."""
        x = torch.linspace(-0.99, 0.99, 50)
        result = inverse_erf(x)
        expected = torch.special.erfinv(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
