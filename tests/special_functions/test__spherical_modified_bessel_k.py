import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_modified_bessel_k


class TestSphericalModifiedBesselK(BinaryOperatorTestCase):
    func = staticmethod(spherical_modified_bessel_k)
    op_name = "_spherical_modified_bessel_k"

    known_values = []  # Has singularity at x=0

    reference = None

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (0.1, 5.0)  # x (positive)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_k0(self):
        """Test k_0(x) = (pi/(2x)) * exp(-x)."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_modified_bessel_k(n, x)
        expected = (math.pi / (2 * x)) * torch.exp(-x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_k1(self):
        """Test k_1(x) = (pi/(2x)) * exp(-x) * (1 + 1/x)."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_modified_bessel_k(n, x)
        expected = (math.pi / (2 * x)) * torch.exp(-x) * (1 + 1 / x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_positive(self):
        """Test k_n(x) > 0 for x > 0."""
        n = torch.linspace(0.0, 3.0, 10)
        x = torch.linspace(0.1, 5.0, 10)
        result = spherical_modified_bessel_k(n, x)
        assert torch.all(result > 0), "Spherical modified Bessel k should be positive"

    def test_decreasing(self):
        """Test k_n(x) is decreasing in x."""
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x1 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x2 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)

        k1 = spherical_modified_bessel_k(n, x1)
        k2 = spherical_modified_bessel_k(n, x2)

        assert torch.all(k1 > k2), "k_n should decrease with x"
