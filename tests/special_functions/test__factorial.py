import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import factorial


class TestFactorial(UnaryOperatorTestCase):
    func = staticmethod(factorial)
    op_name = "_factorial"

    symmetry = None
    period = None
    bounds = None
    lower_bound = 1.0  # n! >= 1 for n >= 0

    known_values = {
        0.0: 1.0,   # 0! = 1
        1.0: 1.0,   # 1! = 1
        2.0: 2.0,   # 2! = 2
        3.0: 6.0,   # 3! = 6
        4.0: 24.0,  # 4! = 24
        5.0: 120.0, # 5! = 120
    }

    reference = staticmethod(lambda n: torch.tensor(
        [math.factorial(int(x)) for x in n.tolist()],
        dtype=n.dtype
    ))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (0.0, 10.0)
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]  # Non-integers for gradient
    extreme_values = [0.0, 5.0, 10.0, 15.0]

    supports_complex = False

    def test_integer_values(self):
        """Test factorial at integer values."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        result = factorial(n)
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gamma_relation(self):
        """Test n! = Gamma(n+1)."""
        from torchscience.special_functions import gamma
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = factorial(n)
        expected = gamma(n + 1)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test n! = n * (n-1)!."""
        n = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        fact_n = factorial(n)
        fact_nm1 = factorial(n - 1)
        expected = n * fact_nm1
        torch.testing.assert_close(fact_n, expected, atol=1e-6, rtol=1e-6)

    def test_increasing(self):
        """Test factorial is increasing for n >= 0."""
        n = torch.linspace(0.0, 10.0, 11)
        result = factorial(n)
        diff = result[1:] - result[:-1]
        assert torch.all(diff >= 0), "Factorial should be increasing"

    def test_large_values(self):
        """Test factorial for larger values."""
        n = torch.tensor([10.0, 12.0], dtype=torch.float64)
        result = factorial(n)
        expected = torch.tensor([3628800.0, 479001600.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-6)
