import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import modified_bessel_k, modified_bessel_k_derivative


class TestModifiedBesselKDerivative(BinaryOperatorTestCase):
    func = staticmethod(modified_bessel_k_derivative)
    op_name = "_modified_bessel_k_derivative"

    known_values = []

    # Reference: scipy.special.kvp
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.kvp(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 5.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_derivative_relation(self):
        """Test K'_nu(x) = -(K_{nu-1}(x) + K_{nu+1}(x)) / 2."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        k_deriv = modified_bessel_k_derivative(nu, x)
        expected = -(modified_bessel_k(nu - 1, x) + modified_bessel_k(nu + 1, x)) / 2

        torch.testing.assert_close(k_deriv, expected, atol=1e-6, rtol=1e-6)

    def test_k0_derivative(self):
        """Test K'_0(x) = -K_1(x)."""
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = modified_bessel_k_derivative(nu, x)
        expected = -modified_bessel_k(torch.ones_like(x), x)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_numerical_derivative(self):
        """Test derivative matches finite difference."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = 1e-6

        numerical = (modified_bessel_k(nu, x + h) - modified_bessel_k(nu, x - h)) / (2 * h)
        result = modified_bessel_k_derivative(nu, x)

        torch.testing.assert_close(result, numerical, atol=1e-5, rtol=1e-5)

    def test_negative_values(self):
        """Test K'_nu(x) < 0 for all x > 0."""
        nu = torch.linspace(0.0, 3.0, 10)
        x = torch.linspace(0.1, 5.0, 10)
        result = modified_bessel_k_derivative(nu, x)
        assert torch.all(result < 0), "K'_nu(x) should be negative"
