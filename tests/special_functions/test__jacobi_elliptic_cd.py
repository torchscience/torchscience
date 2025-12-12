import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_cd


class TestJacobiEllipticCd(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_cd)
    op_name = "_jacobi_elliptic_cd"

    # Known values for cd(u, k) = cn/dn
    known_values = [
        ((0.0, 0.0), 1.0),  # cd(0, k) = 1
        ((0.0, 0.5), 1.0),  # cd(0, k) = 1
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(cn / dn).to(u.dtype)

    # Input ranges
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_cd_at_zero(self):
        """Test cd(0, k) = 1."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.ones(5)
        torch.testing.assert_close(jacobi_elliptic_cd(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_cd_with_k_zero(self):
        """Test cd(u, 0) = cos(u) (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        expected = torch.cos(u)
        torch.testing.assert_close(jacobi_elliptic_cd(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_cd_equals_cn_over_dn(self):
        """Test cd(u, k) = cn(u, k) / dn(u, k)."""
        from torchscience.special_functions import jacobi_elliptic_cn, jacobi_elliptic_dn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        cd = jacobi_elliptic_cd(u, k)
        cn = jacobi_elliptic_cn(u, k)
        dn = jacobi_elliptic_dn(u, k)

        torch.testing.assert_close(cd, cn / dn, atol=1e-5, rtol=1e-5)

    def test_even_function_in_u(self):
        """Test cd(-u, k) = cd(u, k) (even in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_cd(-u, k),
            jacobi_elliptic_cd(u, k),
            atol=1e-5, rtol=1e-5
        )
