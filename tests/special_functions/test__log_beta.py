import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import log_beta, beta


class TestLogBeta(BinaryOperatorTestCase):
    func = staticmethod(log_beta)
    op_name = "_log_beta"

    known_values = [
        ((1.0, 1.0), 0.0),              # log(B(1, 1)) = log(1) = 0
        ((0.5, 0.5), math.log(math.pi)), # log(B(1/2, 1/2)) = log(pi)
    ]

    # Reference: scipy.special.betaln
    reference = staticmethod(lambda a, b: torch.from_numpy(
        scipy.special.betaln(a.numpy(), b.numpy())
    ).to(a.dtype))

    input_range_1 = (0.1, 10.0)
    input_range_2 = (0.1, 10.0)

    gradcheck_inputs = ([1.0, 2.0, 3.0], [1.5, 2.5, 0.5])

    supports_complex = False

    def test_log_of_beta(self):
        """Test log_beta(a, b) = log(beta(a, b))."""
        a = torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0, 2.0, 0.5], dtype=torch.float64)
        result = log_beta(a, b)
        expected = torch.log(beta(a, b))
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_symmetry(self):
        """Test log_beta(a, b) = log_beta(b, a)."""
        a = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5], dtype=torch.float64)
        torch.testing.assert_close(log_beta(a, b), log_beta(b, a), atol=1e-6, rtol=1e-5)

    def test_gamma_relation(self):
        """Test log_beta(a, b) = log_gamma(a) + log_gamma(b) - log_gamma(a+b)."""
        from torchscience.special_functions import log_gamma
        a = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        b = torch.tensor([1.0, 1.0, 2.0, 0.5, 2.5], dtype=torch.float64)
        result = log_beta(a, b)
        expected = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_large_arguments(self):
        """Test log_beta for large arguments (where beta would overflow)."""
        a = torch.tensor([100.0, 200.0], dtype=torch.float64)
        b = torch.tensor([100.0, 200.0], dtype=torch.float64)
        result = log_beta(a, b)
        # Should be finite for large arguments
        assert torch.all(torch.isfinite(result)), "log_beta should be finite for large args"
