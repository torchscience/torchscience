import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import bessel_j


class TestBesselJ(BinaryOperatorTestCase):
    func = staticmethod(bessel_j)
    op_name = "_bessel_j"

    # Known values for J_nu(x)
    known_values = [
        ((0.0, 0.0), 1.0),  # J_0(0) = 1
        ((1.0, 0.0), 0.0),  # J_1(0) = 0
        ((2.0, 0.0), 0.0),  # J_2(0) = 0
    ]

    # Reference: scipy.special.jv
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.jv(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    # Input ranges: nu can be any real, x typically non-negative for real results
    input_range_1 = (0.0, 5.0)  # nu (order)
    input_range_2 = (0.1, 10.0)  # x (argument)

    # Gradcheck inputs (positive values, avoid x=0 for numerical stability)
    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    # Bessel functions don't have simple complex extensions
    supports_complex = False

    def test_j0_at_origin(self):
        """Test J_0(0) = 1."""
        nu = torch.tensor([0.0])
        x = torch.tensor([0.0])
        expected = torch.tensor([1.0])
        torch.testing.assert_close(bessel_j(nu, x), expected, atol=1e-6, rtol=1e-5)

    def test_jn_at_origin(self):
        """Test J_n(0) = 0 for n > 0."""
        nu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        x = torch.zeros_like(nu)
        expected = torch.zeros_like(nu)
        torch.testing.assert_close(bessel_j(nu, x), expected, atol=1e-6, rtol=1e-5)

    def test_recurrence_relation(self):
        """Test J_{nu-1}(x) + J_{nu+1}(x) = (2*nu/x) * J_nu(x)."""
        nu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = torch.tensor([2.0, 3.0, 4.0, 5.0])

        j_nu_minus_1 = bessel_j(nu - 1, x)
        j_nu = bessel_j(nu, x)
        j_nu_plus_1 = bessel_j(nu + 1, x)

        lhs = j_nu_minus_1 + j_nu_plus_1
        rhs = (2 * nu / x) * j_nu
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_derivative_relation(self):
        """Test 2 * J'_nu(x) = J_{nu-1}(x) - J_{nu+1}(x)."""
        nu = torch.tensor([0.0, 1.0, 2.0])
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        output = bessel_j(nu, x)
        grad = torch.autograd.grad(output.sum(), x)[0]

        j_nu_minus_1 = bessel_j(nu, x - 0.0001)  # Just for verification
        j_nu_minus_1 = bessel_j(nu - 1, x.detach())
        j_nu_plus_1 = bessel_j(nu + 1, x.detach())
        expected_grad = (j_nu_minus_1 - j_nu_plus_1) / 2

        torch.testing.assert_close(grad, expected_grad, atol=1e-4, rtol=1e-4)

    def test_j0_zeros(self):
        """Test that J_0 has zeros approximately at known locations."""
        # First few zeros of J_0: 2.4048, 5.5201, 8.6537
        nu = torch.tensor([0.0, 0.0, 0.0])
        zeros = torch.tensor([2.4048255577, 5.5200781103, 8.6537279129])
        output = bessel_j(nu, zeros)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-5, rtol=1e-5)

    def test_j1_zeros(self):
        """Test that J_1 has zeros approximately at known locations."""
        # First few zeros of J_1: 3.8317, 7.0156, 10.1735
        nu = torch.tensor([1.0, 1.0, 1.0])
        zeros = torch.tensor([3.8317059702, 7.0155866698, 10.1734681351])
        output = bessel_j(nu, zeros)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-5, rtol=1e-5)

    def test_negative_order(self):
        """Test J_{-n}(x) = (-1)^n * J_n(x) for integer n."""
        x = torch.tensor([1.0, 2.0, 3.0])
        for n in [1, 2, 3]:
            nu_pos = torch.full_like(x, float(n))
            nu_neg = torch.full_like(x, float(-n))
            j_pos = bessel_j(nu_pos, x)
            j_neg = bessel_j(nu_neg, x)
            expected = ((-1) ** n) * j_pos
            torch.testing.assert_close(j_neg, expected, atol=1e-5, rtol=1e-5)

    def test_half_integer_order(self):
        """Test Bessel functions at half-integer orders."""
        # J_{1/2}(x) = sqrt(2/(pi*x)) * sin(x)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        nu = torch.full_like(x, 0.5)
        output = bessel_j(nu, x)
        expected = torch.sqrt(2 / (math.pi * x)) * torch.sin(x)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_bounded_output(self):
        """Test |J_nu(x)| <= 1 for nu >= 0."""
        nu = torch.linspace(0.0, 5.0, 20)
        x = torch.linspace(0.1, 20.0, 20)
        output = bessel_j(nu, x)
        assert torch.all(torch.abs(output) <= 1.0 + 1e-6), "Bessel J should be bounded by 1"
