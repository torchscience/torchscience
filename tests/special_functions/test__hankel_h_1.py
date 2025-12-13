import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import hankel_h_1, bessel_j, bessel_y


class TestHankelH1(BinaryOperatorTestCase):
    func = staticmethod(hankel_h_1)
    op_name = "_hankel_h_1"

    known_values = []

    # Reference: scipy.special.hankel1
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.hankel1(nu.numpy(), x.numpy())
    ).to(torch.complex128))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = True

    def test_bessel_relation(self):
        """Test H_1^(1)(x) = J_nu(x) + i*Y_nu(x)."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h1 = hankel_h_1(nu, x)
        j = bessel_j(nu, x)
        y = bessel_y(nu, x)
        expected = torch.complex(j, y)

        torch.testing.assert_close(h1.real, expected.real, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h1.imag, expected.imag, atol=1e-5, rtol=1e-5)

    def test_complex_output(self):
        """Test output is complex."""
        nu = torch.tensor([0.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = hankel_h_1(nu, x)
        assert result.is_complex(), "Hankel function should return complex"

    def test_specific_values(self):
        """Test specific values match scipy."""
        nu = torch.tensor([0.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        result = hankel_h_1(nu, x)
        # H_0^(1)(1) = J_0(1) + i*Y_0(1)
        expected_real = 0.7651976866  # J_0(1)
        expected_imag = 0.0882569642  # Y_0(1)
        torch.testing.assert_close(result.real, torch.tensor([expected_real], dtype=torch.float64), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.imag, torch.tensor([expected_imag], dtype=torch.float64), atol=1e-5, rtol=1e-5)
