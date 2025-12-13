import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import hankel_h_2, bessel_j, bessel_y


class TestHankelH2(BinaryOperatorTestCase):
    func = staticmethod(hankel_h_2)
    op_name = "_hankel_h_2"

    known_values = []

    # Reference: scipy.special.hankel2
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.hankel2(nu.numpy(), x.numpy())
    ).to(torch.complex128))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = True

    def test_bessel_relation(self):
        """Test H_nu^(2)(x) = J_nu(x) - i*Y_nu(x)."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h2 = hankel_h_2(nu, x)
        j = bessel_j(nu, x)
        y = bessel_y(nu, x)
        expected = torch.complex(j, -y)

        torch.testing.assert_close(h2.real, expected.real, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h2.imag, expected.imag, atol=1e-5, rtol=1e-5)

    def test_complex_output(self):
        """Test output is complex."""
        nu = torch.tensor([0.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = hankel_h_2(nu, x)
        assert result.is_complex(), "Hankel function should return complex"

    def test_conjugate_relation(self):
        """Test H_nu^(2)(x) = conj(H_nu^(1)(x)) for real nu, x."""
        from torchscience.special_functions import hankel_h_1
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h1 = hankel_h_1(nu, x)
        h2 = hankel_h_2(nu, x)

        torch.testing.assert_close(h2, torch.conj(h1), atol=1e-5, rtol=1e-5)
