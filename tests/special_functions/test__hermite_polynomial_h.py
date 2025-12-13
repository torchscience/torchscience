import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import hermite_polynomial_h


class TestHermitePolynomialH(BinaryOperatorTestCase):
    func = staticmethod(hermite_polynomial_h)
    op_name = "_hermite_polynomial_h"

    # Physicist's Hermite polynomials
    known_values = [
        ((0.0, 1.0), 1.0),     # H_0(x) = 1
        ((1.0, 1.0), 2.0),     # H_1(x) = 2x
        ((2.0, 1.0), 2.0),     # H_2(x) = 4x^2 - 2
        ((3.0, 1.0), -4.0),    # H_3(x) = 8x^3 - 12x
    ]

    # Reference: scipy.special.eval_hermite
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.eval_hermite(n.numpy().astype(int), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 8.0)  # n
    input_range_2 = (-3.0, 3.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 1.5])

    supports_complex = False

    def test_h0(self):
        """Test H_0(x) = 1."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_h(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_h1(self):
        """Test H_1(x) = 2x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_h(n, x)
        expected = 2 * x
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_h2(self):
        """Test H_2(x) = 4x^2 - 2."""
        n = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_h(n, x)
        expected = 4 * x ** 2 - 2
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        h_np1 = hermite_polynomial_h(n + 1, x)
        h_n = hermite_polynomial_h(n, x)
        h_nm1 = hermite_polynomial_h(n - 1, x)

        expected = 2 * x * h_n - 2 * n * h_nm1
        torch.testing.assert_close(h_np1, expected, atol=1e-4, rtol=1e-4)

    def test_at_zero(self):
        """Test H_n(0) values."""
        # H_{2n}(0) = (-1)^n * (2n)! / n!, H_{2n+1}(0) = 0
        n_odd = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        x_zero = torch.zeros_like(n_odd)
        result_odd = hermite_polynomial_h(n_odd, x_zero)
        torch.testing.assert_close(result_odd, torch.zeros_like(result_odd), atol=1e-6, rtol=1e-6)
