import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import riemann_zeta


class TestRiemannZeta(UnaryOperatorTestCase):
    func = staticmethod(riemann_zeta)
    op_name = "_riemann_zeta"

    symmetry = None
    period = None
    bounds = None

    known_values = {
        2.0: math.pi ** 2 / 6,          # zeta(2) = pi^2/6
        4.0: math.pi ** 4 / 90,         # zeta(4) = pi^4/90
        6.0: math.pi ** 6 / 945,        # zeta(6) = pi^6/945
    }

    # Reference: scipy.special.zeta
    reference = staticmethod(lambda s: torch.from_numpy(
        scipy.special.zeta(s.numpy(), 1)
    ).to(s.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (1.1, 10.0)  # zeta has pole at s=1
    gradcheck_inputs = [2.0, 3.0, 4.0, 5.0]
    extreme_values = [1.5, 2.0, 5.0, 10.0]

    supports_complex = False

    def test_zeta_2(self):
        """Test zeta(2) = pi^2/6."""
        s = torch.tensor([2.0], dtype=torch.float64)
        result = riemann_zeta(s)
        expected = torch.tensor([math.pi ** 2 / 6], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_zeta_4(self):
        """Test zeta(4) = pi^4/90."""
        s = torch.tensor([4.0], dtype=torch.float64)
        result = riemann_zeta(s)
        expected = torch.tensor([math.pi ** 4 / 90], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_specific_values(self):
        """Test specific values."""
        test_cases = [
            (3.0, 1.2020569032),  # zeta(3) - Apery's constant
            (5.0, 1.0369277551),  # zeta(5)
        ]
        for s_val, expected_val in test_cases:
            s = torch.tensor([s_val], dtype=torch.float64)
            result = riemann_zeta(s)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_decreasing(self):
        """Test zeta(s) is decreasing for s > 1."""
        s = torch.linspace(1.5, 10.0, 20)
        result = riemann_zeta(s)
        diff = result[1:] - result[:-1]
        assert torch.all(diff < 1e-6), "Zeta should be decreasing for s > 1"

    def test_limit(self):
        """Test zeta(s) -> 1 as s -> infinity."""
        s = torch.tensor([20.0, 50.0], dtype=torch.float64)
        result = riemann_zeta(s)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_greater_than_one(self):
        """Test zeta(s) > 1 for s > 1."""
        s = torch.linspace(1.1, 10.0, 20)
        result = riemann_zeta(s)
        assert torch.all(result > 1.0), "Zeta should be > 1 for s > 1"
