import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_sc


class TestJacobiEllipticSc(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_sc)
    op_name = "_jacobi_elliptic_sc"

    # Known values for sc(u, k) = sn/cn
    known_values = [
        ((0.0, 0.0), 0.0),  # sc(0, k) = 0
        ((0.0, 0.5), 0.0),  # sc(0, k) = 0
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(sn / cn).to(u.dtype)

    # Input ranges - avoid values where cn is small
    input_range_1 = (-1.0, 1.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 0.7, 0.9], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_sc_at_zero(self):
        """Test sc(0, k) = 0."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.zeros(5)
        torch.testing.assert_close(jacobi_elliptic_sc(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_sc_with_k_zero(self):
        """Test sc(u, 0) = tan(u) (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0])
        k = torch.zeros_like(u)
        expected = torch.tan(u)
        torch.testing.assert_close(jacobi_elliptic_sc(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_sc_equals_sn_over_cn(self):
        """Test sc(u, k) = sn(u, k) / cn(u, k)."""
        from torchscience.special_functions import jacobi_elliptic_sn, jacobi_elliptic_cn

        u = torch.tensor([0.5, 0.7, 0.9, 1.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        sc = jacobi_elliptic_sc(u, k)
        sn = jacobi_elliptic_sn(u, k)
        cn = jacobi_elliptic_cn(u, k)

        torch.testing.assert_close(sc, sn / cn, atol=1e-5, rtol=1e-5)

    def test_odd_function_in_u(self):
        """Test sc(-u, k) = -sc(u, k) (odd in u)."""
        u = torch.tensor([0.5, 0.7, 0.9, 1.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_sc(-u, k),
            -jacobi_elliptic_sc(u, k),
            atol=1e-5, rtol=1e-5
        )
