import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import kelvin_kei


class TestKelvinKei(BinaryOperatorTestCase):
    func = staticmethod(kelvin_kei)
    op_name = "_kelvin_kei"

    known_values = []

    reference = None

    input_range_1 = (0.0, 3.0)  # v (order)
    input_range_2 = (0.1, 10.0)  # x

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_kei0_specific_values(self):
        """Test kei_0 at specific values against scipy."""
        v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = kelvin_kei(v, x)
        # Values from scipy.special.kei
        expected = torch.tensor([
            scipy.special.kei(1.0),
            scipy.special.kei(2.0),
            scipy.special.kei(3.0)
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
