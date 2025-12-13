import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import double_factorial


class TestDoubleFactorial(UnaryOperatorTestCase):
    func = staticmethod(double_factorial)
    op_name = "_double_factorial"

    symmetry = None
    period = None
    bounds = None
    lower_bound = 1.0

    # n!! = n * (n-2) * (n-4) * ... * 1 or 2
    known_values = {
        0.0: 1.0,    # 0!! = 1
        1.0: 1.0,    # 1!! = 1
        2.0: 2.0,    # 2!! = 2
        3.0: 3.0,    # 3!! = 3
        4.0: 8.0,    # 4!! = 4*2 = 8
        5.0: 15.0,   # 5!! = 5*3*1 = 15
        6.0: 48.0,   # 6!! = 6*4*2 = 48
        7.0: 105.0,  # 7!! = 7*5*3*1 = 105
    }

    reference = None
    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (0.0, 10.0)
    gradcheck_inputs = [0.5, 1.5, 2.5, 3.5]
    extreme_values = [0.0, 5.0, 10.0]

    supports_complex = False

    def test_integer_values(self):
        """Test double factorial at integer values."""
        n = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        result = double_factorial(n)
        expected = torch.tensor([1.0, 1.0, 2.0, 3.0, 8.0, 15.0, 48.0, 105.0, 384.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test n!! = n * (n-2)!!."""
        n = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        dfact_n = double_factorial(n)
        dfact_nm2 = double_factorial(n - 2)
        expected = n * dfact_nm2
        torch.testing.assert_close(dfact_n, expected, atol=1e-6, rtol=1e-6)

    def test_positive(self):
        """Test double factorial is positive."""
        n = torch.linspace(0.0, 10.0, 20)
        result = double_factorial(n)
        assert torch.all(result > 0), "Double factorial should be positive"

    def test_even_odd_relation(self):
        """Test (2n)!! = 2^n * n!."""
        from torchscience.special_functions import factorial
        n = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = double_factorial(2 * n)
        expected = (2 ** n) * factorial(n)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
