import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_modified_bessel_i


class TestSphericalModifiedBesselI(BinaryOperatorTestCase):
    func = staticmethod(spherical_modified_bessel_i)
    op_name = "_spherical_modified_bessel_i"

    known_values = [
        ((0.0, 0.0), 1.0),  # i_0(0) = 1
    ]

    reference = None

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (0.0, 5.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_i0(self):
        """Test i_0(x) = sinh(x)/x."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_modified_bessel_i(n, x)
        expected = torch.sinh(x) / x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_i1(self):
        """Test i_1(x) = cosh(x)/x - sinh(x)/x^2."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_modified_bessel_i(n, x)
        expected = torch.cosh(x) / x - torch.sinh(x) / (x ** 2)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_at_origin(self):
        """Test i_0(0) = 1."""
        n = torch.tensor([0.0], dtype=torch.float64)
        x = torch.tensor([0.0], dtype=torch.float64)
        result = spherical_modified_bessel_i(n, x)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_positive(self):
        """Test i_n(x) > 0 for x > 0."""
        n = torch.linspace(0.0, 3.0, 10)
        x = torch.linspace(0.1, 5.0, 10)
        result = spherical_modified_bessel_i(n, x)
        assert torch.all(result > 0), "Spherical modified Bessel i should be positive"
