import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_elliptic_dn


class TestJacobiEllipticDn(BinaryOperatorTestCase):
    func = staticmethod(jacobi_elliptic_dn)
    op_name = "_jacobi_elliptic_dn"

    # Known values for dn(u, k)
    known_values = [
        ((0.0, 0.0), 1.0),  # dn(0, k) = 1
        ((0.0, 0.5), 1.0),  # dn(0, k) = 1
        ((0.0, 1.0), 1.0),  # dn(0, k) = 1
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph)
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(dn).to(u.dtype)

    # Input ranges
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_dn_at_zero(self):
        """Test dn(0, k) = 1."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.ones(5)
        torch.testing.assert_close(jacobi_elliptic_dn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_dn_with_k_zero(self):
        """Test dn(u, 0) = 1 (trigonometric case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        expected = torch.ones_like(u)
        torch.testing.assert_close(jacobi_elliptic_dn(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_second_pythagorean_identity(self):
        """Test k^2 * sn^2(u, k) + dn^2(u, k) = 1."""
        from torchscience.special_functions import jacobi_elliptic_sn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        sn = jacobi_elliptic_sn(u, k)
        dn = jacobi_elliptic_dn(u, k)

        torch.testing.assert_close(k ** 2 * sn ** 2 + dn ** 2, torch.ones_like(sn), atol=1e-5, rtol=1e-5)

    def test_even_function_in_u(self):
        """Test dn(-u, k) = dn(u, k) (even in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_elliptic_dn(-u, k),
            jacobi_elliptic_dn(u, k),
            atol=1e-5, rtol=1e-5
        )

    def test_bounded_output(self):
        """Test sqrt(1 - k^2) <= dn(u, k) <= 1."""
        u = torch.linspace(-5.0, 5.0, 20)
        k = torch.full_like(u, 0.5)
        output = jacobi_elliptic_dn(u, k)
        lower_bound = torch.sqrt(1 - k ** 2)
        assert torch.all(output >= lower_bound[0] - 1e-6), "dn should be >= sqrt(1-k^2)"
        assert torch.all(output <= 1.0 + 1e-6), "dn should be <= 1"
