import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import carlson_elliptic_integral_r_f


class TestCarlsonEllipticIntegralRF(BinaryOperatorTestCase):
    func = staticmethod(lambda x, y: carlson_elliptic_integral_r_f(x, y, torch.ones_like(x)))
    op_name = "_carlson_elliptic_integral_r_f"

    known_values = []

    reference = None

    input_range_1 = (0.1, 5.0)  # x
    input_range_2 = (0.1, 5.0)  # y

    gradcheck_inputs = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    supports_complex = False


class TestCarlsonEllipticIntegralRF3(BinaryOperatorTestCase):
    """Test full 3-argument R_F(x, y, z)."""

    @staticmethod
    def func(x, y):
        z = torch.ones_like(x)
        return carlson_elliptic_integral_r_f(x, y, z)

    op_name = "_carlson_elliptic_integral_r_f"

    known_values = []

    reference = None

    input_range_1 = (0.1, 5.0)
    input_range_2 = (0.1, 5.0)

    gradcheck_inputs = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_symmetry(self):
        """Test R_F is symmetric in its arguments."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([3.0, 4.0], dtype=torch.float64)

        rf_xyz = carlson_elliptic_integral_r_f(x, y, z)
        rf_yxz = carlson_elliptic_integral_r_f(y, x, z)
        rf_zyx = carlson_elliptic_integral_r_f(z, y, x)

        torch.testing.assert_close(rf_xyz, rf_yxz, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(rf_xyz, rf_zyx, atol=1e-6, rtol=1e-6)

    def test_homogeneity(self):
        """Test R_F(tx, ty, tz) = t^(-1/2) * R_F(x, y, z)."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([2.0, 3.0], dtype=torch.float64)
        z = torch.tensor([3.0, 4.0], dtype=torch.float64)
        t = 4.0

        rf_scaled = carlson_elliptic_integral_r_f(t * x, t * y, t * z)
        rf_original = carlson_elliptic_integral_r_f(x, y, z)
        expected = rf_original / math.sqrt(t)

        torch.testing.assert_close(rf_scaled, expected, atol=1e-6, rtol=1e-6)

    def test_equal_arguments(self):
        """Test R_F(x, x, x) = 1/sqrt(x)."""
        x = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        result = carlson_elliptic_integral_r_f(x, x, x)
        expected = 1.0 / torch.sqrt(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_positive(self):
        """Test R_F is positive for positive arguments."""
        x = torch.linspace(0.1, 5.0, 10)
        y = torch.linspace(0.1, 5.0, 10)
        z = torch.ones_like(x)
        result = carlson_elliptic_integral_r_f(x, y, z)
        assert torch.all(result > 0), "R_F should be positive"
