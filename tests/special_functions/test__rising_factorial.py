import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import rising_factorial


class TestRisingFactorial(BinaryOperatorTestCase):
    func = staticmethod(rising_factorial)
    op_name = "_rising_factorial"

    # Rising factorial (Pochhammer symbol): (x)_n = x(x+1)(x+2)...(x+n-1)
    known_values = [
        ((1.0, 0.0), 1.0),   # (1)_0 = 1
        ((1.0, 1.0), 1.0),   # (1)_1 = 1
        ((1.0, 2.0), 2.0),   # (1)_2 = 1*2 = 2
        ((1.0, 3.0), 6.0),   # (1)_3 = 1*2*3 = 6
        ((2.0, 3.0), 24.0),  # (2)_3 = 2*3*4 = 24
        ((3.0, 2.0), 12.0),  # (3)_2 = 3*4 = 12
    ]

    # Reference: scipy.special.poch
    reference = staticmethod(lambda x, n: torch.from_numpy(
        scipy.special.poch(x.numpy(), n.numpy())
    ).to(x.dtype))

    input_range_1 = (0.1, 5.0)  # x
    input_range_2 = (0.0, 5.0)  # n

    gradcheck_inputs = ([1.0, 2.0, 3.0], [1.0, 2.0, 2.0])

    supports_complex = False

    def test_n_zero(self):
        """Test (x)_0 = 1."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        n = torch.zeros_like(x)
        result = rising_factorial(x, n)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_n_one(self):
        """Test (x)_1 = x."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        n = torch.ones_like(x)
        result = rising_factorial(x, n)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_integer_x_n(self):
        """Test specific integer values."""
        test_cases = [
            (1.0, 4.0, 24.0),   # (1)_4 = 1*2*3*4 = 24 = 4!
            (2.0, 4.0, 120.0),  # (2)_4 = 2*3*4*5 = 120
            (5.0, 3.0, 210.0),  # (5)_3 = 5*6*7 = 210
        ]
        for x_val, n_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            n = torch.tensor([n_val], dtype=torch.float64)
            result = rising_factorial(x, n)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_factorial_relation(self):
        """Test (1)_n = n!."""
        from torchscience.special_functions import factorial
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        x = torch.ones_like(n)
        result = rising_factorial(x, n)
        expected = factorial(n)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gamma_relation(self):
        """Test (x)_n = Gamma(x+n) / Gamma(x)."""
        from torchscience.special_functions import gamma
        x = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        n = torch.tensor([2.0, 3.0, 2.0], dtype=torch.float64)
        result = rising_factorial(x, n)
        expected = gamma(x + n) / gamma(x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
