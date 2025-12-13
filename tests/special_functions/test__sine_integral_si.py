import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import sine_integral_si


class TestSineIntegralSi(UnaryOperatorTestCase):
    func = staticmethod(sine_integral_si)
    op_name = "_sine_integral_si"

    symmetry = "odd"  # Si(-x) = -Si(x)
    period = None
    bounds = None

    known_values = {
        0.0: 0.0,  # Si(0) = 0
    }

    # Reference: scipy.special.sici returns (si, ci)
    reference = staticmethod(lambda x: torch.from_numpy(
        scipy.special.sici(x.numpy())[0]
    ).to(x.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 10.0)
    gradcheck_inputs = [0.5, 1.0, 2.0, 3.0]
    extreme_values = [-5.0, 0.0, 5.0, 10.0]

    def test_at_zero(self):
        """Test Si(0) = 0."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = sine_integral_si(x)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_odd_symmetry(self):
        """Test Si(-x) = -Si(x)."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        si_pos = sine_integral_si(x)
        si_neg = sine_integral_si(-x)
        torch.testing.assert_close(si_neg, -si_pos, atol=1e-6, rtol=1e-6)

    def test_specific_values(self):
        """Test specific values of Si."""
        test_cases = [
            (1.0, 0.9460831),   # Si(1)
            (2.0, 1.6054129),   # Si(2)
            (math.pi, 1.8519370),  # Si(pi)
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = sine_integral_si(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_limit(self):
        """Test Si(x) -> pi/2 as x -> infinity."""
        x = torch.tensor([50.0, 100.0], dtype=torch.float64)
        result = sine_integral_si(x)
        expected = torch.full_like(result, math.pi / 2)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)
