import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import bessel_j, bessel_j_derivative


class TestBesselJDerivative(BinaryOperatorTestCase):
    func = staticmethod(bessel_j_derivative)
    op_name = "_bessel_j_derivative"

    known_values = []

    # Reference: scipy.special.jvp
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.jvp(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_derivative_relation(self):
        """Test J'_nu(x) = (J_{nu-1}(x) - J_{nu+1}(x)) / 2."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        j_deriv = bessel_j_derivative(nu, x)
        expected = (bessel_j(nu - 1, x) - bessel_j(nu + 1, x)) / 2

        torch.testing.assert_close(j_deriv, expected, atol=1e-6, rtol=1e-6)

    def test_j0_derivative(self):
        """Test J'_0(x) = -J_1(x)."""
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = bessel_j_derivative(nu, x)
        expected = -bessel_j(torch.ones_like(x), x)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_numerical_derivative(self):
        """Test derivative matches finite difference."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = 1e-6

        numerical = (bessel_j(nu, x + h) - bessel_j(nu, x - h)) / (2 * h)
        result = bessel_j_derivative(nu, x)

        torch.testing.assert_close(result, numerical, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values of J'_nu(x)."""
        test_cases = [
            (0.0, 1.0, -0.4400505857),  # J'_0(1)
            (1.0, 1.0, 0.3251471008),   # J'_1(1)
            (0.0, 2.0, -0.5767248078),  # J'_0(2)
        ]
        for nu_val, x_val, expected_val in test_cases:
            nu = torch.tensor([nu_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = bessel_j_derivative(nu, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_at_origin(self):
        """Test J'_nu(0) = 0 for nu > 1, 1/2 for nu = 1, undefined for nu = 0."""
        # For nu = 1: J'_1(0) = 1/2
        nu = torch.tensor([1.0], dtype=torch.float64)
        x = torch.tensor([1e-10], dtype=torch.float64)  # Near zero
        result = bessel_j_derivative(nu, x)
        expected = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
