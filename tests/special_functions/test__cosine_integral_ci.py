import math

import torch
import scipy.special

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import cosine_integral_ci


class TestCosineIntegralCi(UnaryOperatorTestCase):
    func = staticmethod(cosine_integral_ci)
    op_name = "_cosine_integral_ci"

    symmetry = None  # Ci is only defined for x > 0
    period = None
    bounds = None

    known_values = {}

    # Reference: scipy.special.sici returns (si, ci)
    reference = staticmethod(lambda x: torch.from_numpy(
        scipy.special.sici(x.numpy())[1]
    ).to(x.dtype))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (0.1, 10.0)  # Ci defined for x > 0
    gradcheck_inputs = [0.5, 1.0, 2.0, 3.0]
    extreme_values = [0.1, 1.0, 5.0, 10.0]

    def test_specific_values(self):
        """Test specific values of Ci."""
        test_cases = [
            (1.0, 0.3374039),   # Ci(1)
            (2.0, 0.4229808),   # Ci(2)
            (math.pi, 0.0736679),  # Ci(pi)
        ]
        for x_val, expected_val in test_cases:
            x = torch.tensor([x_val], dtype=torch.float64)
            result = cosine_integral_ci(x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_limit(self):
        """Test Ci(x) -> 0 as x -> infinity."""
        x = torch.tensor([50.0, 100.0], dtype=torch.float64)
        result = cosine_integral_ci(x)
        torch.testing.assert_close(result, torch.zeros_like(result), atol=0.1, rtol=0.5)

    def test_oscillation(self):
        """Test Ci oscillates around 0."""
        x = torch.linspace(1.0, 20.0, 100)
        result = cosine_integral_ci(x)
        # Check that there are sign changes
        sign_changes = torch.sum(result[1:] * result[:-1] < 0)
        assert sign_changes > 0, "Ci should oscillate"
