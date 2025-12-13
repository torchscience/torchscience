import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import legendre_q


class TestLegendreQ(BinaryOperatorTestCase):
    func = staticmethod(legendre_q)
    op_name = "_legendre_q"

    known_values = []

    reference = None

    input_range_1 = (0.0, 5.0)  # n
    input_range_2 = (-0.99, 0.99)  # x (|x| < 1 for Q to be real)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_q0(self):
        """Test Q_0(x) = (1/2) * ln((1+x)/(1-x))."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        result = legendre_q(n, x)
        expected = 0.5 * torch.log((1 + x) / (1 - x))
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_q1(self):
        """Test Q_1(x) = x*Q_0(x) - 1."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        q0 = legendre_q(torch.zeros_like(n), x)
        result = legendre_q(n, x)
        expected = x * q0 - 1
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test (n+1)*Q_{n+1}(x) = (2n+1)*x*Q_n(x) - n*Q_{n-1}(x)."""
        n = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

        q_np1 = legendre_q(n + 1, x)
        q_n = legendre_q(n, x)
        q_nm1 = legendre_q(n - 1, x)

        lhs = (n + 1) * q_np1
        rhs = (2 * n + 1) * x * q_n - n * q_nm1
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_at_zero(self):
        """Test Q_n(0) for even/odd n."""
        # Q_n(0) = 0 for even n (Q_0(0) is exception)
        # Actually Q_0(0) = 0 since ln(1) = 0
        n = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
        x = torch.zeros_like(n)
        result = legendre_q(n, x)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=1e-6)
