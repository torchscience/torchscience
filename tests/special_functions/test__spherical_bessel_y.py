import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import spherical_bessel_y


class TestSphericalBesselY(BinaryOperatorTestCase):
    func = staticmethod(spherical_bessel_y)
    op_name = "_spherical_bessel_y"

    known_values = []  # Has singularity at x=0

    # Reference: scipy.special.spherical_yn
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.spherical_yn(n.numpy().astype(int), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (0.1, 10.0)  # x (positive)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_y0(self):
        """Test y_0(x) = -cos(x)/x."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_bessel_y(n, x)
        expected = -torch.cos(x) / x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_y1(self):
        """Test y_1(x) = -cos(x)/x^2 - sin(x)/x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = spherical_bessel_y(n, x)
        expected = -torch.cos(x) / (x ** 2) - torch.sin(x) / x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test recurrence: (2n+1)/x * y_n(x) = y_{n-1}(x) + y_{n+1}(x)."""
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        y_n = spherical_bessel_y(n, x)
        y_nm1 = spherical_bessel_y(n - 1, x)
        y_np1 = spherical_bessel_y(n + 1, x)

        lhs = (2 * n + 1) / x * y_n
        rhs = y_nm1 + y_np1
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values."""
        test_cases = [
            (0.0, 1.0, -0.5403023059),   # y_0(1) = -cos(1)/1
            (0.0, math.pi, 1.0 / math.pi),  # y_0(pi) = -cos(pi)/pi = 1/pi
        ]
        for n_val, x_val, expected_val in test_cases:
            n = torch.tensor([n_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = spherical_bessel_y(n, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
