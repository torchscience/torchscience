import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_hankel_h_2, spherical_bessel_j, spherical_bessel_y


class TestSphericalHankelH2(BinaryOperatorTestCase):
    func = staticmethod(spherical_hankel_h_2)
    op_name = "_spherical_hankel_h_2"

    known_values = []

    reference = None

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = True

    def test_bessel_relation(self):
        """Test h_n^(2)(x) = j_n(x) - i*y_n(x)."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h2 = spherical_hankel_h_2(n, x)
        j = spherical_bessel_j(n, x)
        y = spherical_bessel_y(n, x)
        expected = torch.complex(j, -y)

        torch.testing.assert_close(h2.real, expected.real, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h2.imag, expected.imag, atol=1e-5, rtol=1e-5)

    def test_h0(self):
        """Test h_0^(2)(x) = i * exp(-ix) / x."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = spherical_hankel_h_2(n, x)
        expected = 1j * torch.exp(-1j * x) / x

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_conjugate_relation(self):
        """Test h_n^(2)(x) = conj(h_n^(1)(x))."""
        from torchscience.special_functions import spherical_hankel_h_1
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        h1 = spherical_hankel_h_1(n, x)
        h2 = spherical_hankel_h_2(n, x)

        torch.testing.assert_close(h2, torch.conj(h1), atol=1e-5, rtol=1e-5)
