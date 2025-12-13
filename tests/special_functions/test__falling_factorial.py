import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import falling_factorial


class TestFallingFactorial(BinaryOperatorTestCase):
    func = staticmethod(falling_factorial)
    op_name = "_falling_factorial"

    # Falling factorial: (x)_n = x(x-1)(x-2)...(x-n+1)
    known_values = [
        ((5.0, 0.0), 1.0),   # (5)_0 = 1
        ((5.0, 1.0), 5.0),   # (5)_1 = 5
        ((5.0, 2.0), 20.0),  # (5)_2 = 5*4 = 20
        ((5.0, 3.0), 60.0),  # (5)_3 = 5*4*3 = 60
        ((5.0, 4.0), 120.0), # (5)_4 = 5*4*3*2 = 120
        ((5.0, 5.0), 120.0), # (5)_5 = 5! = 120
    ]

    reference = None

    input_range_1 = (1.0, 10.0)  # x
    input_range_2 = (0.0, 5.0)   # n

    gradcheck_inputs = ([3.0, 4.0, 5.0], [1.0, 2.0, 2.0])

    supports_complex = False

    def test_n_zero(self):
        """Test (x)_0 = 1."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        n = torch.zeros_like(x)
        result = falling_factorial(x, n)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_n_one(self):
        """Test (x)_1 = x."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        n = torch.ones_like(x)
        result = falling_factorial(x, n)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_factorial_relation(self):
        """Test (n)_n = n!."""
        from torchscience.special_functions import factorial
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        result = falling_factorial(n, n)
        expected = factorial(n)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_binomial_relation(self):
        """Test (n)_k / k! = C(n, k)."""
        from torchscience.special_functions import factorial, binomial_coefficient
        n = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        k = torch.tensor([2.0, 3.0, 3.0, 4.0], dtype=torch.float64)
        result = falling_factorial(n, k) / factorial(k)
        expected = binomial_coefficient(n, k)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific falling factorial values."""
        test_cases = [
            (6.0, 3.0, 120.0),  # 6*5*4 = 120
            (4.0, 2.0, 12.0),   # 4*3 = 12
            (10.0, 2.0, 90.0),  # 10*9 = 90
        ]
        for x_val, n_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            n = torch.tensor([n_val], dtype=torch.float64)
            result = falling_factorial(x, n)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
