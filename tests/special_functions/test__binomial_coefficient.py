import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import binomial_coefficient


class TestBinomialCoefficient(BinaryOperatorTestCase):
    func = staticmethod(binomial_coefficient)
    op_name = "_binomial_coefficient"

    known_values = [
        ((5.0, 0.0), 1.0),   # C(5,0) = 1
        ((5.0, 1.0), 5.0),   # C(5,1) = 5
        ((5.0, 2.0), 10.0),  # C(5,2) = 10
        ((5.0, 3.0), 10.0),  # C(5,3) = 10
        ((5.0, 4.0), 5.0),   # C(5,4) = 5
        ((5.0, 5.0), 1.0),   # C(5,5) = 1
        ((10.0, 5.0), 252.0), # C(10,5) = 252
    ]

    reference = staticmethod(lambda n, k: torch.from_numpy(
        scipy.special.comb(n.numpy(), k.numpy(), exact=False)
    ).to(n.dtype))

    input_range_1 = (0.0, 15.0)  # n
    input_range_2 = (0.0, 10.0)  # k

    gradcheck_inputs = ([5.0, 6.0, 7.0], [2.0, 3.0, 3.0])

    supports_complex = False

    def test_symmetry(self):
        """Test C(n, k) = C(n, n-k)."""
        n = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        k = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float64)
        result_k = binomial_coefficient(n, k)
        result_nk = binomial_coefficient(n, n - k)
        torch.testing.assert_close(result_k, result_nk, atol=1e-6, rtol=1e-6)

    def test_boundary_k_zero(self):
        """Test C(n, 0) = 1."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64)
        k = torch.zeros_like(n)
        result = binomial_coefficient(n, k)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_boundary_k_n(self):
        """Test C(n, n) = 1."""
        n = torch.tensor([0.0, 1.0, 5.0, 10.0], dtype=torch.float64)
        result = binomial_coefficient(n, n)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_pascals_triangle(self):
        """Test Pascal's identity: C(n, k) = C(n-1, k-1) + C(n-1, k)."""
        n = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        k = torch.tensor([2.0, 3.0, 3.0, 4.0], dtype=torch.float64)
        result = binomial_coefficient(n, k)
        expected = binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_specific_values(self):
        """Test specific binomial coefficients."""
        test_cases = [
            (4.0, 2.0, 6.0),    # C(4,2) = 6
            (6.0, 3.0, 20.0),   # C(6,3) = 20
            (10.0, 4.0, 210.0), # C(10,4) = 210
        ]
        for n_val, k_val, expected_val in test_cases:
            n = torch.tensor([n_val], dtype=torch.float64)
            k = torch.tensor([k_val], dtype=torch.float64)
            result = binomial_coefficient(n, k)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_positive_integers(self):
        """Test binomial coefficients are positive integers for non-negative integer inputs."""
        n = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float64)
        k = torch.tensor([2.0, 3.0, 2.0, 4.0, 3.0], dtype=torch.float64)
        result = binomial_coefficient(n, k)
        assert torch.all(result > 0), "Binomial coefficients should be positive"
        # Check they are close to integers
        torch.testing.assert_close(result, torch.round(result), atol=1e-6, rtol=1e-6)
