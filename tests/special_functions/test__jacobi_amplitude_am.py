import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import jacobi_amplitude_am


class TestJacobiAmplitudeAm(BinaryOperatorTestCase):
    func = staticmethod(jacobi_amplitude_am)
    op_name = "_jacobi_amplitude_am"

    # Known values for am(u, k)
    known_values = [
        ((0.0, 0.0), 0.0),  # am(0, k) = 0
        ((0.0, 0.5), 0.0),  # am(0, k) = 0
        ((0.0, 1.0), 0.0),  # am(0, k) = 0
        ((1.0, 0.0), 1.0),  # am(u, 0) = u (when k=0)
    ]

    # Reference: scipy.special.ellipj returns (sn, cn, dn, ph) where ph = am
    @staticmethod
    def reference(u, k):
        sn, cn, dn, ph = scipy.special.ellipj(u.numpy(), k.numpy() ** 2)
        return torch.from_numpy(ph).to(u.dtype)

    # Input ranges: u can be any real, k in [0, 1]
    input_range_1 = (-5.0, 5.0)  # u (argument)
    input_range_2 = (0.0, 0.99)  # k (modulus), avoid k=1 for numerical stability

    # Gradcheck inputs
    gradcheck_inputs = ([0.5, 1.0, 1.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_am_at_zero(self):
        """Test am(0, k) = 0."""
        u = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.99])
        expected = torch.zeros(5)
        torch.testing.assert_close(jacobi_amplitude_am(u, k), expected, atol=1e-6, rtol=1e-5)

    def test_am_with_k_zero(self):
        """Test am(u, 0) = u (linear case)."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        k = torch.zeros_like(u)
        torch.testing.assert_close(jacobi_amplitude_am(u, k), u, atol=1e-6, rtol=1e-5)

    def test_sn_cn_dn_relationship(self):
        """Test sn(u,k) = sin(am(u,k)), cn(u,k) = cos(am(u,k))."""
        from torchscience.special_functions import jacobi_elliptic_sn, jacobi_elliptic_cn

        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])

        am = jacobi_amplitude_am(u, k)
        sn = jacobi_elliptic_sn(u, k)
        cn = jacobi_elliptic_cn(u, k)

        torch.testing.assert_close(torch.sin(am), sn, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.cos(am), cn, atol=1e-5, rtol=1e-5)

    def test_odd_function_in_u(self):
        """Test am(-u, k) = -am(u, k) (odd in u)."""
        u = torch.tensor([0.5, 1.0, 1.5, 2.0])
        k = torch.tensor([0.3, 0.4, 0.5, 0.6])
        torch.testing.assert_close(
            jacobi_amplitude_am(-u, k),
            -jacobi_amplitude_am(u, k),
            atol=1e-5, rtol=1e-5
        )
