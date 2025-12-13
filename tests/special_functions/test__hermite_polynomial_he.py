import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import hermite_polynomial_he


class TestHermitePolynomialHe(BinaryOperatorTestCase):
    func = staticmethod(hermite_polynomial_he)
    op_name = "_hermite_polynomial_he"

    # Probabilist's Hermite polynomials
    known_values = [
        ((0.0, 1.0), 1.0),     # He_0(x) = 1
        ((1.0, 1.0), 1.0),     # He_1(x) = x
        ((2.0, 1.0), 0.0),     # He_2(x) = x^2 - 1
        ((3.0, 1.0), -2.0),    # He_3(x) = x^3 - 3x
    ]

    # Reference: scipy.special.eval_hermitenorm
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.eval_hermitenorm(n.numpy().astype(int), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 8.0)  # n
    input_range_2 = (-3.0, 3.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 1.5])

    supports_complex = False

    def test_he0(self):
        """Test He_0(x) = 1."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_he(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_he1(self):
        """Test He_1(x) = x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_he(n, x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_he2(self):
        """Test He_2(x) = x^2 - 1."""
        n = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result = hermite_polynomial_he(n, x)
        expected = x ** 2 - 1
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        he_np1 = hermite_polynomial_he(n + 1, x)
        he_n = hermite_polynomial_he(n, x)
        he_nm1 = hermite_polynomial_he(n - 1, x)

        expected = x * he_n - n * he_nm1
        torch.testing.assert_close(he_np1, expected, atol=1e-4, rtol=1e-4)

    def test_relation_to_h(self):
        """Test He_n(x) = 2^(-n/2) * H_n(x/sqrt(2))."""
        from torchscience.special_functions import hermite_polynomial_h
        n = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)

        he = hermite_polynomial_he(n, x)
        h = hermite_polynomial_h(n, x / math.sqrt(2))
        expected = h / (2 ** (n / 2))
        torch.testing.assert_close(he, expected, atol=1e-5, rtol=1e-5)
