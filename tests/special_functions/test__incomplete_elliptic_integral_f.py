import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import incomplete_elliptic_integral_f, complete_elliptic_integral_k


class TestIncompleteEllipticIntegralF(BinaryOperatorTestCase):
    func = staticmethod(incomplete_elliptic_integral_f)
    op_name = "_incomplete_elliptic_integral_f"

    known_values = [
        ((0.0, 0.5), 0.0),  # F(0, k) = 0
    ]

    # Reference: scipy.special.ellipkinc
    reference = staticmethod(lambda phi, k: torch.from_numpy(
        scipy.special.ellipkinc(phi.numpy(), k.numpy())
    ).to(phi.dtype))

    input_range_1 = (0.0, math.pi / 2 - 0.1)  # phi (amplitude)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    gradcheck_inputs = ([0.3, 0.5, 1.0], [0.3, 0.5, 0.7])

    supports_complex = False

    def test_at_zero_amplitude(self):
        """Test F(0, k) = 0."""
        phi = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64)
        result = incomplete_elliptic_integral_f(phi, k)
        expected = torch.zeros_like(result)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_complete_integral(self):
        """Test F(pi/2, k) = K(k)."""
        phi = torch.full((3,), math.pi / 2, dtype=torch.float64)
        k = torch.tensor([0.0, 0.5, 0.8], dtype=torch.float64)
        result = incomplete_elliptic_integral_f(phi, k)
        expected = complete_elliptic_integral_k(k)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_k_zero(self):
        """Test F(phi, 0) = phi."""
        phi = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
        k = torch.zeros_like(phi)
        result = incomplete_elliptic_integral_f(phi, k)
        torch.testing.assert_close(result, phi, atol=1e-6, rtol=1e-6)

    def test_increasing_in_phi(self):
        """Test F is increasing in phi."""
        phi = torch.linspace(0.0, math.pi / 2 - 0.1, 20)
        k = torch.full_like(phi, 0.5)
        result = incomplete_elliptic_integral_f(phi, k)
        diff = result[1:] - result[:-1]
        assert torch.all(diff >= -1e-6), "F should be increasing in phi"

    def test_specific_values(self):
        """Test specific known values."""
        test_cases = [
            (0.5, 0.5, 0.5153882032),  # F(0.5, 0.5)
            (1.0, 0.5, 1.0895506700),  # F(1.0, 0.5)
        ]
        for phi_val, k_val, expected_val in test_cases:
            phi = torch.tensor([phi_val], dtype=torch.float64)
            k = torch.tensor([k_val], dtype=torch.float64)
            result = incomplete_elliptic_integral_f(phi, k)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
