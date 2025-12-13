import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_hankel_h_1, spherical_bessel_j, spherical_bessel_y


class TestSphericalHankelH1(BinaryOperatorTestCase):
    func = staticmethod(spherical_hankel_h_1)
    op_name = "_spherical_hankel_h_1"

    known_values = []

    reference = None

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = True

    def test_bessel_relation(self):
        """Test h_n^(1)(x) = j_n(x) + i*y_n(x)."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h1 = spherical_hankel_h_1(n, x)
        j = spherical_bessel_j(n, x)
        y = spherical_bessel_y(n, x)
        expected = torch.complex(j, y)

        torch.testing.assert_close(h1.real, expected.real, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h1.imag, expected.imag, atol=1e-5, rtol=1e-5)

    def test_h0(self):
        """Test h_0^(1)(x) = -i * exp(ix) / x."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = spherical_hankel_h_1(n, x)
        # h_0^(1)(x) = j_0(x) + i*y_0(x) = sin(x)/x - i*cos(x)/x = -i*exp(ix)/x
        expected = -1j * torch.exp(1j * x) / x

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
