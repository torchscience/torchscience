import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import legendre_p


class TestLegendreP(BinaryOperatorTestCase):
    func = staticmethod(legendre_p)
    op_name = "_legendre_p"

    known_values = [
        ((0.0, 0.5), 1.0),     # P_0(x) = 1
        ((1.0, 0.5), 0.5),     # P_1(x) = x
        ((2.0, 0.5), -0.125),  # P_2(x) = (3x^2 - 1)/2
    ]

    # Reference: scipy.special.eval_legendre
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.eval_legendre(n.numpy(), x.numpy())
    ).to(n.dtype))

    input_range_1 = (0.0, 10.0)  # n
    input_range_2 = (-1.0, 1.0)   # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_p0(self):
        """Test P_0(x) = 1."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = legendre_p(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_p1(self):
        """Test P_1(x) = x."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = legendre_p(n, x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_p2(self):
        """Test P_2(x) = (3x^2 - 1)/2."""
        n = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
        result = legendre_p(n, x)
        expected = (3 * x ** 2 - 1) / 2
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        p_np1 = legendre_p(n + 1, x)
        p_n = legendre_p(n, x)
        p_nm1 = legendre_p(n - 1, x)

        lhs = (n + 1) * p_np1
        rhs = (2 * n + 1) * x * p_n - n * p_nm1
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_at_one(self):
        """Test P_n(1) = 1."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.ones_like(n)
        result = legendre_p(n, x)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_at_minus_one(self):
        """Test P_n(-1) = (-1)^n."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = -torch.ones_like(n)
        result = legendre_p(n, x)
        expected = torch.pow(-1.0, n)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_bounded(self):
        """Test |P_n(x)| <= 1 for x in [-1, 1]."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        x = torch.linspace(-1.0, 1.0, 5)
        result = legendre_p(n, x)
        assert torch.all(torch.abs(result) <= 1.0 + 1e-6), "P_n should be bounded by 1"

    def test_orthogonality_integral(self):
        """Test orthogonality at specific points (not full integral)."""
        # P_n(0) = 0 for odd n
        n_odd = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        x_zero = torch.zeros_like(n_odd)
        result = legendre_p(n_odd, x_zero)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=1e-6)
