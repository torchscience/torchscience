import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import kelvin_bei


class TestKelvinBei(BinaryOperatorTestCase):
    func = staticmethod(kelvin_bei)
    op_name = "_kelvin_bei"

    known_values = [
        ((0.0, 0.0), 0.0),  # bei_0(0) = 0
    ]

    reference = None

    input_range_1 = (0.0, 3.0)  # v (order)
    input_range_2 = (0.0, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_bei0_at_zero(self):
        """Test bei_0(0) = 0."""
        v = torch.tensor([0.0], dtype=torch.float64)
        x = torch.tensor([0.0], dtype=torch.float64)
        result = kelvin_bei(v, x)
        expected = torch.tensor([0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_bei0_specific_values(self):
        """Test bei_0 at specific values against scipy."""
        v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = kelvin_bei(v, x)
        # Values from scipy.special.bei
        expected = torch.tensor([
            scipy.special.bei(1.0),
            scipy.special.bei(2.0),
            scipy.special.bei(3.0)
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
