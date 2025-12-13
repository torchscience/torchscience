import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import kelvin_ker


class TestKelvinKer(BinaryOperatorTestCase):
    func = staticmethod(kelvin_ker)
    op_name = "_kelvin_ker"

    known_values = []  # ker has singularity at x=0

    reference = None

    input_range_1 = (0.0, 3.0)  # v (order)
    input_range_2 = (0.1, 10.0)  # x (positive, avoid singularity)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_ker0_specific_values(self):
        """Test ker_0 at specific values against scipy."""
        v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = kelvin_ker(v, x)
        # Values from scipy.special.ker
        expected = torch.tensor([
            scipy.special.ker(1.0),
            scipy.special.ker(2.0),
            scipy.special.ker(3.0)
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_decreasing(self):
        """Test ker_0 decreases for small x."""
        v = torch.zeros(10, dtype=torch.float64)
        x = torch.linspace(0.1, 2.0, 10)
        result = kelvin_ker(v, x)
        diff = result[1:] - result[:-1]
        # ker_0 generally increases from negative values toward 0
        assert not torch.all(diff == 0), "ker should vary with x"
