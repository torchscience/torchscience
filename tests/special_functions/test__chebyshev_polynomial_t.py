import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import chebyshev_polynomial_t


class TestChebyshevPolynomialT(BinaryOperatorTestCase):
    func = staticmethod(chebyshev_polynomial_t)
    op_name = "_chebyshev_polynomial_t"

    # T_n(cos(theta)) = cos(n*theta)
    known_values = [
        ((0.0, 0.5), 1.0),    # T_0(x) = 1
        ((1.0, 0.5), 0.5),    # T_1(x) = x
        ((2.0, 0.5), -0.5),   # T_2(x) = 2x^2 - 1
        ((3.0, 0.5), -1.0),   # T_3(x) = 4x^3 - 3x
    ]

    # Reference: scipy.special.eval_chebyt
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.eval_chebyt(n.numpy(), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 10.0)  # n (degree)
    input_range_2 = (-1.0, 1.0)   # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_t0(self):
        """Test T_0(x) = 1."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_t(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_t1(self):
        """Test T_1(x) = x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_t(n, x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_t2(self):
        """Test T_2(x) = 2x^2 - 1."""
        n = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = chebyshev_polynomial_t(n, x)
        expected = 2 * x ** 2 - 1
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        t_np1 = chebyshev_polynomial_t(n + 1, x)
        t_n = chebyshev_polynomial_t(n, x)
        t_nm1 = chebyshev_polynomial_t(n - 1, x)

        expected = 2 * x * t_n - t_nm1
        torch.testing.assert_close(t_np1, expected, atol=1e-5, rtol=1e-5)

    def test_cosine_relation(self):
        """Test T_n(cos(theta)) = cos(n*theta)."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        theta = torch.tensor([0.5, 0.7, 0.3, 0.9], dtype=torch.float64)
        x = torch.cos(theta)

        result = chebyshev_polynomial_t(n, x)
        expected = torch.cos(n * theta)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_at_one(self):
        """Test T_n(1) = 1."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones_like(n)
        result = chebyshev_polynomial_t(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_at_minus_one(self):
        """Test T_n(-1) = (-1)^n."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = -torch.ones_like(n)
        result = chebyshev_polynomial_t(n, x)
        expected = torch.pow(-1.0, n)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_bounded(self):
        """Test |T_n(x)| <= 1 for x in [-1, 1]."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        x = torch.linspace(-1.0, 1.0, 5)
        result = chebyshev_polynomial_t(n, x)
        assert torch.all(torch.abs(result) <= 1.0 + 1e-6), "T_n should be bounded by 1"
