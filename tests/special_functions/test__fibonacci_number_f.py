import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import fibonacci_number_f


class TestFibonacciNumberF(UnaryOperatorTestCase):
    func = staticmethod(fibonacci_number_f)
    op_name = "_fibonacci_number_f"

    symmetry = None
    period = None
    bounds = None
    lower_bound = 0.0

    # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    known_values = {
        0.0: 0.0,
        1.0: 1.0,
        2.0: 1.0,
        3.0: 2.0,
        4.0: 3.0,
        5.0: 5.0,
        6.0: 8.0,
        7.0: 13.0,
        8.0: 21.0,
        9.0: 34.0,
        10.0: 55.0,
    }

    reference = None

    input_range = (0.0, 20.0)
    gradcheck_inputs = [1.5, 2.5, 3.5, 4.5]  # Non-integers for gradient
    extreme_values = [0.0, 5.0, 10.0, 15.0]

    supports_complex = False

    def test_f0(self):
        """Test F_0 = 0."""
        n = torch.tensor([0.0], dtype=torch.float64)
        result = fibonacci_number_f(n)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_f1(self):
        """Test F_1 = 1."""
        n = torch.tensor([1.0], dtype=torch.float64)
        result = fibonacci_number_f(n)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test F_n = F_{n-1} + F_{n-2}."""
        n = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float64)
        f_n = fibonacci_number_f(n)
        f_nm1 = fibonacci_number_f(n - 1)
        f_nm2 = fibonacci_number_f(n - 2)
        expected = f_nm1 + f_nm2
        torch.testing.assert_close(f_n, expected, atol=1e-5, rtol=1e-5)

    def test_sequence(self):
        """Test first several Fibonacci numbers."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float64)
        result = fibonacci_number_f(n)
        expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_binet_formula(self):
        """Test Binet's formula: F_n = (phi^n - psi^n) / sqrt(5)."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        psi = (1 - math.sqrt(5)) / 2
        n = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float64)
        result = fibonacci_number_f(n)
        expected = (phi ** n - psi ** n) / math.sqrt(5)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_nonnegative(self):
        """Test F_n >= 0 for n >= 0."""
        n = torch.linspace(0.0, 20.0, 21)
        result = fibonacci_number_f(n)
        assert torch.all(result >= 0), "Fibonacci numbers should be non-negative"

    def test_increasing(self):
        """Test Fibonacci sequence is increasing for n >= 1."""
        n = torch.linspace(1.0, 15.0, 15)
        result = fibonacci_number_f(n)
        diff = result[1:] - result[:-1]
        assert torch.all(diff >= 0), "Fibonacci should be non-decreasing"
