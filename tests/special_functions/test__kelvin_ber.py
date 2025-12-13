import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import kelvin_ber


class TestKelvinBer(BinaryOperatorTestCase):
    func = staticmethod(kelvin_ber)
    op_name = "_kelvin_ber"

    known_values = [
        ((0.0, 0.0), 1.0),  # ber_0(0) = 1
    ]

    # Reference: scipy.special.ber for order 0
    reference = None  # scipy only has order 0

    input_range_1 = (0.0, 3.0)  # v (order)
    input_range_2 = (0.0, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_ber0_at_zero(self):
        """Test ber_0(0) = 1."""
        v = torch.tensor([0.0], dtype=torch.float64)
        x = torch.tensor([0.0], dtype=torch.float64)
        result = kelvin_ber(v, x)
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_ber0_specific_values(self):
        """Test ber_0 at specific values against scipy."""
        v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = kelvin_ber(v, x)
        # Values from scipy.special.ber
        expected = torch.tensor([
            scipy.special.ber(1.0),
            scipy.special.ber(2.0),
            scipy.special.ber(3.0)
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_oscillation(self):
        """Test ber oscillates for large x."""
        v = torch.tensor([0.0] * 50, dtype=torch.float64)
        x = torch.linspace(0.1, 20.0, 50)
        result = kelvin_ber(v, x)
        # Check that there are sign changes (oscillation)
        sign_changes = torch.sum(result[1:] * result[:-1] < 0)
        assert sign_changes > 0, "ber should oscillate"
