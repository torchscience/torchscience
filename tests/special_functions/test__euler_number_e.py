import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import euler_number_e


class TestEulerNumberE(UnaryOperatorTestCase):
    func = staticmethod(euler_number_e)
    op_name = "_euler_number_e"

    # Euler numbers have no symmetry (defined only for non-negative integers)
    symmetry = None
    period = None

    # Known values for Euler numbers
    # E_0 = 1, E_1 = 0, E_2 = -1, E_3 = 0, E_4 = 5, E_5 = 0, E_6 = -61
    known_values = {
        0.0: 1.0,
        1.0: 0.0,
        2.0: -1.0,
        3.0: 0.0,
        4.0: 5.0,
        5.0: 0.0,
        6.0: -61.0,
    }

    # No standard reference implementation
    reference = None

    # Input range (non-negative integers)
    input_range = (0.0, 10.0)

    # Gradcheck inputs
    gradcheck_inputs = [0.0, 2.0, 4.0, 6.0]

    # Euler numbers don't support complex inputs meaningfully
    supports_complex = False

    def test_odd_numbers_are_zero(self):
        """Test that E_n = 0 for all odd n."""
        n = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        output = euler_number_e(n)
        expected = torch.zeros_like(n)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_even_numbers_alternate_sign(self):
        """Test that even Euler numbers alternate in sign."""
        # E_0 = 1 (positive)
        # E_2 = -1 (negative)
        # E_4 = 5 (positive)
        # E_6 = -61 (negative)
        # E_8 = 1385 (positive)
        n = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
        output = euler_number_e(n)
        signs = torch.sign(output)
        expected_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        torch.testing.assert_close(signs, expected_signs, atol=1e-6, rtol=1e-5)

    def test_e8(self):
        """Test E_8 = 1385."""
        n = torch.tensor([8.0])
        output = euler_number_e(n)
        expected = torch.tensor([1385.0])
        torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-5)

    def test_e10(self):
        """Test E_10 = -50521."""
        n = torch.tensor([10.0])
        output = euler_number_e(n)
        expected = torch.tensor([-50521.0])
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-5)
