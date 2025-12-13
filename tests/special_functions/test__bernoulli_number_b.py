import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import bernoulli_number_b


class TestBernoulliNumberB(UnaryOperatorTestCase):
    func = staticmethod(bernoulli_number_b)
    op_name = "_bernoulli_number_b"

    symmetry = None
    period = None
    bounds = None

    # Bernoulli numbers
    known_values = {
        0.0: 1.0,       # B_0 = 1
        1.0: -0.5,      # B_1 = -1/2
        2.0: 1.0/6.0,   # B_2 = 1/6
        4.0: -1.0/30.0, # B_4 = -1/30
        6.0: 1.0/42.0,  # B_6 = 1/42
        8.0: -1.0/30.0, # B_8 = -1/30
    }

    reference = None

    input_range = (0.0, 20.0)
    gradcheck_inputs = [0.0, 2.0, 4.0, 6.0]
    extreme_values = [0.0, 4.0, 8.0, 12.0]

    supports_complex = False

    def test_b0(self):
        """Test B_0 = 1."""
        n = torch.tensor([0.0], dtype=torch.float64)
        result = bernoulli_number_b(n)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_b1(self):
        """Test B_1 = -1/2."""
        n = torch.tensor([1.0], dtype=torch.float64)
        result = bernoulli_number_b(n)
        expected = torch.tensor([-0.5], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_odd_zeros(self):
        """Test B_n = 0 for odd n > 1."""
        n = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float64)
        result = bernoulli_number_b(n)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_even_values(self):
        """Test B_n for even n."""
        test_cases = [
            (2.0, 1.0 / 6.0),
            (4.0, -1.0 / 30.0),
            (6.0, 1.0 / 42.0),
            (8.0, -1.0 / 30.0),
            (10.0, 5.0 / 66.0),
        ]
        for n_val, expected_val in test_cases:
            n = torch.tensor([n_val], dtype=torch.float64)
            result = bernoulli_number_b(n)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_alternating_sign(self):
        """Test B_{2n} alternates in sign for n >= 1."""
        n_even = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float64)
        result = bernoulli_number_b(n_even)
        # Signs should alternate
        signs = torch.sign(result)
        expected_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(signs, expected_signs, atol=0, rtol=0)
