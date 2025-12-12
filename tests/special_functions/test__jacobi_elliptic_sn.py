import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_sn


class TestJacobiEllipticSn(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_sn)
    op_name = "_jacobi_elliptic_sn"

    # Known values for sn(u, k)
    known_values = [
        ((0.0, 0.0), 0.0),  # sn(0, k) = 0
        ((0.0, 0.5), 0.0),  # sn(0, k) = 0
        ((0.0, 1.0), 0.0),  # sn(0, k) = 0
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(sn).to(u.dtype)

    # Input ranges: u can be any real, k in [0, 1]
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_sn_at_zero(self):
        """Test sn(0, k) = 0."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.zeros(5)
        torch.testing.assert_close(jacobi_elliptic_sn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_sn_with_k_zero(self):
        """Test sn(u, 0) = sin(u) (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        expected = torch.sin(u)
        torch.testing.assert_close(jacobi_elliptic_sn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_pythagorean_identity(self):
        """Test sn^2(u, k) + cn^2(u, k) = 1."""
        from torchscience.special_functions import jacobi_elliptic_cn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        sn = jacobi_elliptic_sn(u, k)
        cn = jacobi_elliptic_cn(u, k)

        torch.testing.assert_close(sn ** 2 + cn ** 2, torch.ones_like(sn), atol=1e-5, rtol=1e-5)

    def test_odd_function_in_u(self):
        """Test sn(-u, k) = -sn(u, k) (odd in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_sn(-u, k),
            -jacobi_elliptic_sn(u, k),
            atol=1e-5, rtol=1e-5
        )

    def test_bounded_output(self):
        """Test |sn(u, k)| <= 1."""
        u = torch.linspace(-5.0, 5.0, 20)
        k = torch.full_like(u, 0.5)
        output = jacobi_elliptic_sn(u, k)
        assert torch.all(torch.abs(output) <= 1.0 + 1e-6), "sn should be bounded by 1"
