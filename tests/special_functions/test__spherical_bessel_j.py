import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_bessel_j


class TestSphericalBesselJ(BinaryOperatorTestCase):
    func = staticmethod(spherical_bessel_j)
    op_name = "_spherical_bessel_j"

    known_values = [
        ((0.0, 0.0), 1.0),  # j_0(0) = 1
    ]

    # Reference: scipy.special.spherical_jn
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.spherical_jn(n.numpy().astype(int), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 5.0)  # n (order)
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_j0(self):
        """Test j_0(x) = sin(x)/x."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_bessel_j(n, x)
        expected = torch.sin(x) / x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_j1(self):
        """Test j_1(x) = sin(x)/x^2 - cos(x)/x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_bessel_j(n, x)
        expected = torch.sin(x) / (x ** 2) - torch.cos(x) / x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_at_origin(self):
        """Test j_0(0) = 1 and j_n(0) = 0 for n > 0."""
        # j_0(0) = 1
        n0 = torch.tensor([0.0], dtype=torch.float64)
        x0 = torch.tensor([0.0], dtype=torch.float64)
        result0 = spherical_bessel_j(n0, x0)
        torch.testing.assert_close(result0, torch.tensor([1.0], dtype=torch.float64), atol=1e-6, rtol=1e-6)

        # j_n(0) = 0 for n > 0 (test near zero)
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1e-10, 1e-10, 1e-10], dtype=torch.float64)
        result = spherical_bessel_j(n, x)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-5, rtol=1e-5)

    def test_recurrence(self):
        """Test recurrence: (2n+1)/x * j_n(x) = j_{n-1}(x) + j_{n+1}(x)."""
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        j_n = spherical_bessel_j(n, x)
        j_nm1 = spherical_bessel_j(n - 1, x)
        j_np1 = spherical_bessel_j(n + 1, x)

        lhs = (2 * n + 1) / x * j_n
        rhs = j_nm1 + j_np1
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_bounded(self):
        """Test |j_n(x)| <= 1."""
        n = torch.linspace(0.0, 5.0, 10)
        x = torch.linspace(0.1, 10.0, 10)
        result = spherical_bessel_j(n, x)
        assert torch.all(torch.abs(result) <= 1.0 + 1e-6), "Spherical Bessel j should be bounded"
