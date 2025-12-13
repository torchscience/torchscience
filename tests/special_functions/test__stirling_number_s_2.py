import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import stirling_number_s_2


class TestStirlingNumberS2(BinaryOperatorTestCase):
    func = staticmethod(stirling_number_s_2)
    op_name = "_stirling_number_s_2"

    # Stirling numbers of the second kind
    # S(n, k) = number of ways to partition n elements into k non-empty subsets
    known_values = [
        ((0.0, 0.0), 1.0),   # S(0, 0) = 1
        ((1.0, 1.0), 1.0),   # S(1, 1) = 1
        ((2.0, 1.0), 1.0),   # S(2, 1) = 1
        ((2.0, 2.0), 1.0),   # S(2, 2) = 1
        ((3.0, 1.0), 1.0),   # S(3, 1) = 1
        ((3.0, 2.0), 3.0),   # S(3, 2) = 3
        ((3.0, 3.0), 1.0),   # S(3, 3) = 1
        ((4.0, 1.0), 1.0),   # S(4, 1) = 1
        ((4.0, 2.0), 7.0),   # S(4, 2) = 7
        ((4.0, 3.0), 6.0),   # S(4, 3) = 6
        ((4.0, 4.0), 1.0),   # S(4, 4) = 1
        ((5.0, 2.0), 15.0),  # S(5, 2) = 15
    ]

    reference = None

    input_range_1 = (0.0, 8.0)  # n
    input_range_2 = (0.0, 8.0)  # k

    gradcheck_inputs = ([3.0, 4.0, 5.0], [1.0, 2.0, 2.0])

    supports_complex = False

    def test_boundary_k_zero(self):
        """Test S(n, 0) = 0 for n > 0."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        k = torch.zeros_like(n)
        result = stirling_number_s_2(n, k)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_boundary_k_n(self):
        """Test S(n, n) = 1."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = stirling_number_s_2(n, n)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_boundary_k_one(self):
        """Test S(n, 1) = 1 for n >= 1."""
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        k = torch.ones_like(n)
        result = stirling_number_s_2(n, k)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test S(n+1, k) = k*S(n, k) + S(n, k-1)."""
        n = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        k = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)

        s_np1_k = stirling_number_s_2(n + 1, k)
        s_n_k = stirling_number_s_2(n, k)
        s_n_km1 = stirling_number_s_2(n, k - 1)

        expected = k * s_n_k + s_n_km1
        torch.testing.assert_close(s_np1_k, expected, atol=1e-4, rtol=1e-4)

    def test_bell_number_relation(self):
        """Test sum of S(n, k) over k gives Bell number."""
        # B_4 = S(4,1) + S(4,2) + S(4,3) + S(4,4) = 1 + 7 + 6 + 1 = 15
        n = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = stirling_number_s_2(n, k)
        bell_4 = result.sum()
        expected = torch.tensor(15.0, dtype=torch.float64)
        torch.testing.assert_close(bell_4, expected, atol=1e-5, rtol=1e-5)

    def test_nonnegative(self):
        """Test Stirling numbers are non-negative."""
        n = torch.tensor([3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
        result = stirling_number_s_2(n, k)
        assert torch.all(result >= 0), "Stirling numbers should be non-negative"
