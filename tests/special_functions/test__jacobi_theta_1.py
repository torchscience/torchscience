import math

import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_theta_1


class TestJacobiTheta1(BinaryOperatorTestCase):
    func = staticmethod(jacobi_theta_1)
    op_name = "_jacobi_theta_1"

    known_values = [
        ((0.0, 0.5), 0.0),  # theta_1(0, q) = 0
    ]

    reference = None

    input_range_1 = (-math.pi, math.pi)  # z
    input_range_2 = (0.01, 0.9)  # q (|q| < 1)

    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_at_zero(self):
        """Test theta_1(0, q) = 0."""
        z = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        q = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        result = jacobi_theta_1(z, q)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_odd_in_z(self):
        """Test theta_1(-z, q) = -theta_1(z, q)."""
        z = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        q = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        result_pos = jacobi_theta_1(z, q)
        result_neg = jacobi_theta_1(-z, q)
        torch.testing.assert_close(result_neg, -result_pos, atol=1e-6, rtol=1e-6)

    def test_periodicity(self):
        """Test theta_1(z + pi, q) = -theta_1(z, q)."""
        z = torch.tensor([0.5, 1.0], dtype=torch.float64)
        q = torch.tensor([0.3, 0.5], dtype=torch.float64)
        result = jacobi_theta_1(z, q)
        result_shifted = jacobi_theta_1(z + math.pi, q)
        torch.testing.assert_close(result_shifted, -result, atol=1e-5, rtol=1e-5)

    def test_small_q(self):
        """Test behavior for small q."""
        # For small q, theta_1(z, q) ≈ 2*q^(1/4)*sin(z)
        z = torch.tensor([0.5, 1.0], dtype=torch.float64)
        q = torch.tensor([0.01, 0.01], dtype=torch.float64)
        result = jacobi_theta_1(z, q)
        approx = 2 * q ** 0.25 * torch.sin(z)
        torch.testing.assert_close(result, approx, atol=0.1, rtol=0.1)
