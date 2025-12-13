import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import modified_bessel_i, modified_bessel_i_derivative


class TestModifiedBesselIDerivative(BinaryOperatorTestCase):
    func = staticmethod(modified_bessel_i_derivative)
    op_name = "_modified_bessel_i_derivative"

    known_values = []

    # Reference: scipy.special.ivp
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.ivp(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 5.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_derivative_relation(self):
        """Test I'_nu(x) = (I_{nu-1}(x) + I_{nu+1}(x)) / 2."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        i_deriv = modified_bessel_i_derivative(nu, x)
        expected = (modified_bessel_i(nu - 1, x) + modified_bessel_i(nu + 1, x)) / 2

        torch.testing.assert_close(i_deriv, expected, atol=1e-6, rtol=1e-6)

    def test_i0_derivative(self):
        """Test I'_0(x) = I_1(x)."""
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = modified_bessel_i_derivative(nu, x)
        expected = modified_bessel_i(torch.ones_like(x), x)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_numerical_derivative(self):
        """Test derivative matches finite difference."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = 1e-6

        numerical = (modified_bessel_i(nu, x + h) - modified_bessel_i(nu, x - h)) / (2 * h)
        result = modified_bessel_i_derivative(nu, x)

        torch.testing.assert_close(result, numerical, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values of I'_nu(x)."""
        test_cases = [
            (0.0, 1.0, 0.5651591040),  # I'_0(1) = I_1(1)
            (1.0, 1.0, 0.7009067738),  # I'_1(1)
            (0.0, 2.0, 1.5906368546),  # I'_0(2) = I_1(2)
        ]
        for nu_val, x_val, expected_val in test_cases:
            nu = torch.tensor([nu_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = modified_bessel_i_derivative(nu, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
