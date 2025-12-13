import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import logarithmic_integral_li


class TestLogarithmicIntegralLi(UnaryOperatorTestCase):
    func = staticmethod(logarithmic_integral_li)
    op_name = "_logarithmic_integral_li"

    symmetry = None
    period = None
    bounds = None

    known_values = {
        2.0: 1.0451637801,  # li(2) - Soldner's constant offset
    }

    reference = None

    input_range = (1.1, 10.0)  # li has singularity at x=1
    gradcheck_inputs = [1.5, 2.0, 3.0, 5.0]
    extreme_values = [1.1, 2.0, 5.0, 10.0]

    supports_complex = False

    def test_li_2(self):
        """Test li(2) is approximately 1.045."""
        x = torch.tensor([2.0], dtype=torch.float64)
        result = logarithmic_integral_li(x)
        expected = torch.tensor([1.0451637801], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_increasing(self):
        """Test li(x) is increasing for x > 1."""
        x = torch.linspace(1.5, 10.0, 20)
        result = logarithmic_integral_li(x)
        diff = result[1:] - result[:-1]
        assert torch.all(diff > 0), "li should be increasing for x > 1"

    def test_prime_counting_approximation(self):
        """Test li(x) approximates pi(x) for large x."""
        # li(100) should be close to pi(100) = 25
        x = torch.tensor([100.0], dtype=torch.float64)
        result = logarithmic_integral_li(x)
        # li(100) ≈ 30.126
        assert result[0] > 25, "li(100) should be > 25"
        assert result[0] < 35, "li(100) should be < 35"
