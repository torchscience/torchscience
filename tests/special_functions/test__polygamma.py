import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import polygamma, digamma, trigamma


class TestPolygamma(BinaryOperatorTestCase):
    func = staticmethod(polygamma)
    op_name = "_polygamma"

    known_values = []

    # Reference: scipy.special.polygamma
    reference = staticmethod(lambda n, x: torch.from_numpy(
        scipy.special.polygamma(n.numpy().astype(int), x.numpy())
    ).to(x.dtype))

    input_range_1 = (0.0, 5.0)  # n (order)
    input_range_2 = (0.5, 5.0)  # x (positive)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    supports_complex = False

    def test_psi0_is_digamma(self):
        """Test psi_0(x) = digamma(x)."""
        n = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = polygamma(n, x)
        expected = digamma(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_psi1_is_trigamma(self):
        """Test psi_1(x) = trigamma(x)."""
        n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = polygamma(n, x)
        expected = trigamma(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_specific_values(self):
        """Test specific values."""
        # psi_1(1) = pi^2/6
        n = torch.tensor([1.0], dtype=torch.float64)
        x = torch.tensor([1.0], dtype=torch.float64)
        result = polygamma(n, x)
        expected = torch.tensor([math.pi ** 2 / 6], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_recurrence(self):
        """Test psi_n(x+1) = psi_n(x) + (-1)^n * n! / x^(n+1)."""
        n = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([2.0, 2.0], dtype=torch.float64)

        psi_xp1 = polygamma(n, x + 1)
        psi_x = polygamma(n, x)

        from torchscience.special_functions import factorial
        correction = ((-1) ** n) * factorial(n) / (x ** (n + 1))

        torch.testing.assert_close(psi_xp1, psi_x + correction, atol=1e-5, rtol=1e-5)

    def test_sign_pattern(self):
        """Test psi_n(x) has sign (-1)^(n+1) for n >= 1, x > 0."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # psi_1 (trigamma) is positive
        n1 = torch.ones_like(x)
        result1 = polygamma(n1, x)
        assert torch.all(result1 > 0), "psi_1 should be positive"

        # psi_2 is negative
        n2 = torch.full_like(x, 2.0)
        result2 = polygamma(n2, x)
        assert torch.all(result2 < 0), "psi_2 should be negative"
