import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_sd


class TestJacobiEllipticSd(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_sd)
    op_name = "_jacobi_elliptic_sd"

    # Known values for sd(u, k) = sn/dn
    known_values = [
        ((0.0, 0.0), 0.0),  # sd(0, k) = 0
        ((0.0, 0.5), 0.0),  # sd(0, k) = 0
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(sn / dn).to(u.dtype)

    # Input ranges
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_sd_at_zero(self):
        """Test sd(0, k) = 0."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.zeros(5)
        torch.testing.assert_close(jacobi_elliptic_sd(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_sd_with_k_zero(self):
        """Test sd(u, 0) = sin(u) (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        expected = torch.sin(u)
        torch.testing.assert_close(jacobi_elliptic_sd(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_sd_equals_sn_over_dn(self):
        """Test sd(u, k) = sn(u, k) / dn(u, k)."""
        from torchscience.special_functions import jacobi_elliptic_sn, jacobi_elliptic_dn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        sd = jacobi_elliptic_sd(u, k)
        sn = jacobi_elliptic_sn(u, k)
        dn = jacobi_elliptic_dn(u, k)

        torch.testing.assert_close(sd, sn / dn, atol=1e-5, rtol=1e-5)

    def test_odd_function_in_u(self):
        """Test sd(-u, k) = -sd(u, k) (odd in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_sd(-u, k),
            -jacobi_elliptic_sd(u, k),
            atol=1e-5, rtol=1e-5
        )
