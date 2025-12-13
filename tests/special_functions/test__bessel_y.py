import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import bessel_y, bessel_j


class TestBesselY(BinaryOperatorTestCase):
    func = staticmethod(bessel_y)
    op_name = "_bessel_y"

    # Known values for Y_nu(x)
    known_values = []  # Y has singularity at x=0

    # Reference: scipy.special.yv
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.yv(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    # Input ranges: x must be positive for real results
    input_range_1 = (0.0, 5.0)  # nu (order)
    input_range_2 = (0.1, 10.0)  # x (argument, positive only)

    # Gradcheck inputs (positive x, avoid singularity at x=0)
    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_y0_at_known_values(self):
        """Test Y_0 at known values."""
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        result = bessel_y(nu, x)
        expected = torch.tensor([0.0882569642, 0.5103756726, -0.3085176252], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_y1_at_known_values(self):
        """Test Y_1 at known values."""
        nu = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
        result = bessel_y(nu, x)
        expected = torch.tensor([-0.7812128213, -0.1070324315, 0.1478631434], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_recurrence_relation(self):
        """Test Y_{nu-1}(x) + Y_{nu+1}(x) = (2*nu/x) * Y_nu(x)."""
        nu = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        y_nu_minus_1 = bessel_y(nu - 1, x)
        y_nu = bessel_y(nu, x)
        y_nu_plus_1 = bessel_y(nu + 1, x)

        lhs = y_nu_minus_1 + y_nu_plus_1
        rhs = (2 * nu / x) * y_nu
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_wronskian(self):
        """Test Wronskian: J_nu * Y'_nu - J'_nu * Y_nu = 2/(pi*x)."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = 1e-6

        j_nu = bessel_j(nu, x)
        y_nu = bessel_y(nu, x)

        # Derivatives via finite differences
        j_deriv = (bessel_j(nu, x + h) - bessel_j(nu, x - h)) / (2 * h)
        y_deriv = (bessel_y(nu, x + h) - bessel_y(nu, x - h)) / (2 * h)

        wronskian = j_nu * y_deriv - j_deriv * y_nu
        expected = 2.0 / (math.pi * x)

        torch.testing.assert_close(wronskian, expected, atol=1e-4, rtol=1e-4)

    def test_negative_integer_order(self):
        """Test Y_{-n}(x) = (-1)^n * Y_n(x) for integer n."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        for n in [1, 2, 3]:
            nu_pos = torch.full_like(x, float(n))
            nu_neg = torch.full_like(x, float(-n))
            y_pos = bessel_y(nu_pos, x)
            y_neg = bessel_y(nu_neg, x)
            expected = ((-1) ** n) * y_pos
            torch.testing.assert_close(y_neg, expected, atol=1e-5, rtol=1e-5)

    def test_half_integer_order(self):
        """Test Y_{1/2}(x) = -sqrt(2/(pi*x)) * cos(x)."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        nu = torch.full_like(x, 0.5)
        result = bessel_y(nu, x)
        expected = -torch.sqrt(2 / (math.pi * x)) * torch.cos(x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_y0_zeros(self):
        """Test that Y_0 has zeros approximately at known locations."""
        # First few zeros of Y_0: 0.8936, 3.9577, 7.0861
        nu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        zeros = torch.tensor([0.8935769663, 3.9576784194, 7.0860510603], dtype=torch.float64)
        output = bessel_y(nu, zeros)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-5, rtol=1e-5)
