import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import exponential_integral_ei


class TestExponentialIntegralEi(UnaryOperatorTestCase):
    func = staticmethod(exponential_integral_ei)
    op_name = "_exponential_integral_ei"

    symmetry = None
    period = None
    bounds = None

    known_values = {}

    # Reference: scipy.special.expi
    reference = staticmethod(lambda x: torch.from_numpy(
        scipy.special.expi(x.numpy())
    ).to(x.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (0.1, 10.0)  # Ei(x) is defined for x != 0
    gradcheck_inputs = [0.5, 1.0, 2.0, 3.0]
    extreme_values = [0.1, 1.0, 5.0, 10.0]

    supports_complex = False

    def test_positive_values(self):
        """Test Ei at positive values."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = exponential_integral_ei(x)
        # Ei(1) ≈ 1.8951, Ei(2) ≈ 4.9542
        expected = torch.tensor([1.8951178, 4.9542344, 9.9338326], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_increasing(self):
        """Test Ei(x) is increasing for x > 0."""
        x = torch.linspace(0.1, 5.0, 20)
        result = exponential_integral_ei(x)
        diff = result[1:] - result[:-1]
        assert torch.all(diff > 0), "Ei should be increasing for x > 0"

    def test_asymptotic(self):
        """Test asymptotic behavior: Ei(x) ~ exp(x)/x for large x."""
        x = torch.tensor([10.0, 15.0], dtype=torch.float64)
        result = exponential_integral_ei(x)
        asymptotic = torch.exp(x) / x
        # For large x, Ei(x) ≈ exp(x)/x
        ratio = result / asymptotic
        assert torch.all(ratio > 0.9), "Should approach asymptotic form"
        assert torch.all(ratio < 1.1), "Should approach asymptotic form"
