import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import beta, gamma


class TestBeta(BinaryOperatorTestCase):
    func = staticmethod(beta)
    op_name = "_beta"

    # Known values: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
    known_values = [
        ((1.0, 1.0), 1.0),  # B(1, 1) = 1
        ((1.0, 2.0), 0.5),  # B(1, 2) = 1/2
        ((2.0, 1.0), 0.5),  # B(2, 1) = 1/2 (symmetry)
        ((2.0, 2.0), 1.0 / 6.0),  # B(2, 2) = 1/6
        ((0.5, 0.5), math.pi),  # B(1/2, 1/2) = pi
        ((3.0, 3.0), 1.0 / 30.0),  # B(3, 3) = 1/30
    ]

    # Reference: scipy.special.beta
    reference = staticmethod(lambda a, b: torch.from_numpy(
        scipy.special.beta(a.numpy(), b.numpy())
    ).to(a.dtype))

    # Input ranges: beta requires positive arguments
    input_range_1 = (0.1, 10.0)
    input_range_2 = (0.1, 10.0)

    # Gradcheck inputs (positive values)
    gradcheck_inputs = ([1.0, 2.0, 3.0], [1.5, 2.5, 0.5])

    # Beta does not support complex inputs directly
    supports_complex = False

    # Beta is symmetric: B(a, b) = B(b, a)
    def test_symmetry(self):
        """Test B(a, b) = B(b, a)."""
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])
        torch.testing.assert_close(beta(a, b), beta(b, a), atol=1e-6, rtol=1e-5)

    def test_gamma_relation(self):
        """Test B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)."""
        a = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5])
        b = torch.tensor([1.0, 1.0, 2.0, 0.5, 2.5])
        expected = gamma(a) * gamma(b) / gamma(a + b)
        torch.testing.assert_close(beta(a, b), expected, atol=1e-5, rtol=1e-5)

    def test_recurrence_a(self):
        """Test B(a+1, b) = B(a, b) * a / (a + b)."""
        a = torch.tensor([1.0, 2.0, 3.0, 0.5])
        b = torch.tensor([2.0, 1.5, 1.0, 3.0])
        lhs = beta(a + 1, b)
        rhs = beta(a, b) * a / (a + b)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_recurrence_b(self):
        """Test B(a, b+1) = B(a, b) * b / (a + b)."""
        a = torch.tensor([1.0, 2.0, 3.0, 0.5])
        b = torch.tensor([2.0, 1.5, 1.0, 3.0])
        lhs = beta(a, b + 1)
        rhs = beta(a, b) * b / (a + b)
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_special_values(self):
        """Test special values of the beta function."""
        # B(1, n) = 1/n for positive integer n
        n = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ones = torch.ones_like(n)
        expected = 1.0 / n
        torch.testing.assert_close(beta(ones, n), expected, atol=1e-6, rtol=1e-5)

    def test_positive_output(self):
        """Test beta(a, b) > 0 for positive a, b."""
        a = torch.linspace(0.1, 5.0, 20)
        b = torch.linspace(0.1, 5.0, 20)
        output = beta(a, b)
        assert torch.all(output > 0), "Beta should be positive for positive inputs"
