import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import chebyshev_polynomial_u


class TestChebyshevPolynomialU(BinaryOperatorTestCase):
    func = staticmethod(chebyshev_polynomial_u)
    op_name = "_chebyshev_polynomial_u"

    # U_n(cos(theta)) = sin((n+1)*theta) / sin(theta)
    known_values = [
        ((0.0, 0.5), 1.0),    # U_0(x) = 1
        ((1.0, 0.5), 1.0),    # U_1(x) = 2x
        ((2.0, 0.5), 0.0),    # U_2(x) = 4x^2 - 1
    ]

    # Reference: scipy.special.eval_chebyu
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.eval_chebyu(n.numpy(), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 10.0)  # n
    input_range_2 = (-0.99, 0.99)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_u0(self):
        """Test U_0(x) = 1."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_u(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_u1(self):
        """Test U_1(x) = 2x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_u(n, x)
        expected = 2 * x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_u2(self):
        """Test U_2(x) = 4x^2 - 1."""
        n = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_u(n, x)
        expected = 4 * x ** 2 - 1
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test U_{n+1}(x) = 2x*U_n(x) - U_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        u_np1 = chebyshev_polynomial_u(n + 1, x)
        u_n = chebyshev_polynomial_u(n, x)
        u_nm1 = chebyshev_polynomial_u(n - 1, x)

        expected = 2 * x * u_n - u_nm1
        torch.testing.assert_close(u_np1, expected, atol=1e-5, rtol=1e-5)

    def test_at_one(self):
        """Test U_n(1) = n + 1."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones_like(n)
        result = chebyshev_polynomial_u(n, x)
        expected = n + 1
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
