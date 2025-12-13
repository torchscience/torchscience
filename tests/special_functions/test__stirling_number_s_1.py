import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import stirling_number_s_1


class TestStirlingNumberS1(BinaryOperatorTestCase):
    func = staticmethod(stirling_number_s_1)
    op_name = "_stirling_number_s_1"

    # Unsigned Stirling numbers of the first kind
    # s(n, k) = number of permutations of n elements with k cycles
    known_values = [
        ((0.0, 0.0), 1.0),   # s(0, 0) = 1
        ((1.0, 1.0), 1.0),   # s(1, 1) = 1
        ((2.0, 1.0), 1.0),   # s(2, 1) = 1
        ((2.0, 2.0), 1.0),   # s(2, 2) = 1
        ((3.0, 1.0), 2.0),   # s(3, 1) = 2
        ((3.0, 2.0), 3.0),   # s(3, 2) = 3
        ((3.0, 3.0), 1.0),   # s(3, 3) = 1
        ((4.0, 1.0), 6.0),   # s(4, 1) = 6
        ((4.0, 2.0), 11.0),  # s(4, 2) = 11
        ((4.0, 3.0), 6.0),   # s(4, 3) = 6
        ((4.0, 4.0), 1.0),   # s(4, 4) = 1
    ]

    reference = None

    input_range_1 = (0.0, 8.0)  # n
    input_range_2 = (0.0, 8.0)  # k

    gradcheck_inputs = ([3.0, 4.0, 5.0], [1.0, 2.0, 2.0])

    supports_complex = False

    def test_boundary_k_zero(self):
        """Test s(n, 0) = 0 for n > 0."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        k = torch.zeros_like(n)
        result = stirling_number_s_1(n, k)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_boundary_k_n(self):
        """Test s(n, n) = 1."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = stirling_number_s_1(n, n)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_boundary_k_one(self):
        """Test s(n, 1) = (n-1)!."""
        from torchscience.special_functions import factorial
        n = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        k = torch.ones_like(n)
        result = stirling_number_s_1(n, k)
        expected = factorial(n - 1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_recurrence(self):
        """Test s(n+1, k) = n*s(n, k) + s(n, k-1)."""
        n = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        k = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)

        s_np1_k = stirling_number_s_1(n + 1, k)
        s_n_k = stirling_number_s_1(n, k)
        s_n_km1 = stirling_number_s_1(n, k - 1)

        expected = n * s_n_k + s_n_km1
        torch.testing.assert_close(s_np1_k, expected, atol=1e-4, rtol=1e-4)

    def test_nonnegative(self):
        """Test Stirling numbers are non-negative."""
        n = torch.tensor([3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
        result = stirling_number_s_1(n, k)
        assert torch.all(result >= 0), "Stirling numbers should be non-negative"
