import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import modified_bessel_i


class TestModifiedBesselI(BinaryOperatorTestCase):
    func = staticmethod(modified_bessel_i)
    op_name = "_modified_bessel_i"

    # Known values for I_nu(x)
    known_values = [
        ((0.0, 0.0), 1.0),  # I_0(0) = 1
        ((1.0, 0.0), 0.0),  # I_n(0) = 0 for n > 0
    ]

    # Reference: scipy.special.iv
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.iv(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.0, 5.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_i0_at_origin(self):
        """Test I_0(0) = 1."""
        nu = torch.tensor([0.0], dtype=torch.float64)
        x = torch.tensor([0.0], dtype=torch.float64)
        result = modified_bessel_i(nu, x)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_in_at_origin(self):
        """Test I_n(0) = 0 for n > 0."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        result = modified_bessel_i(nu, x)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence_relation(self):
        """Test I_{nu-1}(x) - I_{nu+1}(x) = (2*nu/x) * I_nu(x)."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        i_nu_minus_1 = modified_bessel_i(nu - 1, x)
        i_nu = modified_bessel_i(nu, x)
        i_nu_plus_1 = modified_bessel_i(nu + 1, x)

        lhs = i_nu_minus_1 - i_nu_plus_1
        rhs = (2 * nu / x) * i_nu
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values of I_nu(x)."""
        test_cases = [
            (0.0, 1.0, 1.2660658777),  # I_0(1)
            (1.0, 1.0, 0.5651591040),  # I_1(1)
            (0.0, 2.0, 2.2795853024),  # I_0(2)
            (1.0, 2.0, 1.5906368546),  # I_1(2)
        ]
        for nu_val, x_val, expected_val in test_cases:
            nu = torch.tensor([nu_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = modified_bessel_i(nu, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_positive_for_positive_x(self):
        """Test I_nu(x) > 0 for all x >= 0, nu >= 0."""
        nu = torch.linspace(0.0, 3.0, 10)
        x = torch.linspace(0.1, 5.0, 10)
        result = modified_bessel_i(nu, x)
        assert torch.all(result > 0), "I_nu(x) should be positive"

    def test_negative_order_relation(self):
        """Test I_{-nu}(x) = I_nu(x) for integer nu."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        for n in [0, 1, 2]:
            nu_pos = torch.full_like(x, float(n))
            nu_neg = torch.full_like(x, float(-n))
            i_pos = modified_bessel_i(nu_pos, x)
            i_neg = modified_bessel_i(nu_neg, x)
            torch.testing.assert_close(i_pos, i_neg, atol=1e-5, rtol=1e-5)

    def test_half_integer_order(self):
        """Test I_{1/2}(x) = sqrt(2/(pi*x)) * sinh(x)."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        nu = torch.full_like(x, 0.5)
        result = modified_bessel_i(nu, x)
        expected = torch.sqrt(2 / (math.pi * x)) * torch.sinh(x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
