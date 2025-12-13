import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import incomplete_elliptic_integral_e, complete_elliptic_integral_e


class TestIncompleteEllipticIntegralE(BinaryOperatorTestCase):
    func = staticmethod(incomplete_elliptic_integral_e)
    op_name = "_incomplete_elliptic_integral_e"

    known_values = [
        ((0.0, 0.5), 0.0),  # E(0, k) = 0
    ]

    # Reference: scipy.special.ellipeinc
    reference = staticmethod(lambda phi, k: torch.from_numpy(
        scipy.special.ellipeinc(phi.numpy(), k.numpy())
    ).to(phi.dtype))

    input_range_1 = (0.0, math.pi / 2 - 0.1)  # phi
    input_range_2 = (0.0, 0.99)  # k

    gradcheck_inputs = ([0.3, 0.5, 1.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_at_zero_amplitude(self):
        """Test E(0, k) = 0."""
        phi = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64)
        result = incomplete_elliptic_integral_e(phi, k)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_complete_integral(self):
        """Test E(pi/2, k) = E(k)."""
        phi = torch.full((3,), math.pi / 2, dtype=torch.float64)
        k = torch.tensor([0.0, 0.5, 0.8], dtype=torch.float64)
        result = incomplete_elliptic_integral_e(phi, k)
        expected = complete_elliptic_integral_e(k)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_k_zero(self):
        """Test E(phi, 0) = phi."""
        phi = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
        k = torch.zeros_like(phi)
        result = incomplete_elliptic_integral_e(phi, k)
        torch.testing.assert_close(result, phi, atol=1e-6, rtol=1e-6)

    def test_increasing_in_phi(self):
        """Test E is increasing in phi."""
        phi = torch.linspace(0.0, math.pi / 2 - 0.1, 20)
        k = torch.full_like(phi, 0.5)
        result = incomplete_elliptic_integral_e(phi, k)
        diff = result[1:] - result[:-1]
        assert torch.all(diff >= -1e-6), "E should be increasing in phi"

    def test_bounded_by_phi(self):
        """Test E(phi, k) <= phi."""
        phi = torch.linspace(0.1, math.pi / 2 - 0.1, 20)
        k = torch.full_like(phi, 0.5)
        result = incomplete_elliptic_integral_e(phi, k)
        assert torch.all(result <= phi + 1e-6), "E should be <= phi"
