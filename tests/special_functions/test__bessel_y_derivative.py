import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import bessel_y, bessel_y_derivative


class TestBesselYDerivative(BinaryOperatorTestCase):
    func = staticmethod(bessel_y_derivative)
    op_name = "_bessel_y_derivative"

    known_values = []

    # Reference: scipy.special.yvp
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.yvp(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 10.0)  # x (positive only)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_derivative_relation(self):
        """Test Y'_nu(x) = (Y_{nu-1}(x) - Y_{nu+1}(x)) / 2."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        y_deriv = bessel_y_derivative(nu, x)
        expected = (bessel_y(nu - 1, x) - bessel_y(nu + 1, x)) / 2

        torch.testing.assert_close(y_deriv, expected, atol=1e-6, rtol=1e-6)

    def test_y0_derivative(self):
        """Test Y'_0(x) = -Y_1(x)."""
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = bessel_y_derivative(nu, x)
        expected = -bessel_y(torch.ones_like(x), x)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_numerical_derivative(self):
        """Test derivative matches finite difference."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = 1e-6

        numerical = (bessel_y(nu, x + h) - bessel_y(nu, x - h)) / (2 * h)
        result = bessel_y_derivative(nu, x)

        torch.testing.assert_close(result, numerical, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values of Y'_nu(x)."""
        test_cases = [
            (0.0, 1.0, 0.7812128213),   # Y'_0(1) = Y_1(1) with sign
            (1.0, 1.0, -0.8694697856),  # Y'_1(1)
            (0.0, 2.0, 0.1070324315),   # Y'_0(2)
        ]
        for nu_val, x_val, expected_val in test_cases:
            nu = torch.tensor([nu_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = bessel_y_derivative(nu, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
