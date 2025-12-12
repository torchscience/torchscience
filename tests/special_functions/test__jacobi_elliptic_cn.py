import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_cn


class TestJacobiEllipticCn(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_cn)
    op_name = "_jacobi_elliptic_cn"

    # Known values for cn(u, k)
    known_values = [
        ((0.0, 0.0), 1.0),  # cn(0, k) = 1
        ((0.0, 0.5), 1.0),  # cn(0, k) = 1
        ((0.0, 1.0), 1.0),  # cn(0, k) = 1
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(cn).to(u.dtype)

    # Input ranges
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_cn_at_zero(self):
        """Test cn(0, k) = 1."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.ones(5)
        torch.testing.assert_close(jacobi_elliptic_cn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_cn_with_k_zero(self):
        """Test cn(u, 0) = cos(u) (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        expected = torch.cos(u)
        torch.testing.assert_close(jacobi_elliptic_cn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_pythagorean_identity(self):
        """Test sn^2(u, k) + cn^2(u, k) = 1."""
        from torchscience.special_functions import jacobi_elliptic_sn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        sn = jacobi_elliptic_sn(u, k)
        cn = jacobi_elliptic_cn(u, k)

        torch.testing.assert_close(sn ** 2 + cn ** 2, torch.ones_like(sn), atol=1e-5, rtol=1e-5)

    def test_even_function_in_u(self):
        """Test cn(-u, k) = cn(u, k) (even in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_cn(-u, k),
            jacobi_elliptic_cn(u, k),
            atol=1e-5, rtol=1e-5
        )

    def test_bounded_output(self):
        """Test |cn(u, k)| <= 1."""
        u = torch.linspace(-5.0, 5.0, 20)
        k = torch.full_like(u, 0.5)
        output = jacobi_elliptic_cn(u, k)
        assert torch.all(torch.abs(output) <= 1.0 + 1e-6), "cn should be bounded by 1"
