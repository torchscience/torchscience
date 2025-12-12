import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import inverse_jacobi_elliptic_dn


class TestInverseJacobiEllipticDn(BinaryOperatorTestCase):
    func = staticmethod(inverse_jacobi_elliptic_dn)
    op_name = "_inverse_jacobi_elliptic_dn"

    # Known values for arcdn(x, k)
    known_values = [
        ((1.0, 0.5), 0.0),  # arcdn(1, k) = 0
    ]

    # Input ranges: x in (sqrt(1-k^2), 1), k in (0, 1)
    input_range_1 = (0.9, 1.0)  # x (must be > sqrt(1-k^2))
    input_range_2 = (0.1, 0.3)   # k (small modulus to ensure valid range)

    # Gradcheck inputs - need careful range selection
    gradcheck_inputs = ([0.95, 0.97, 0.99], [0.1, 0.15, 0.2])

    # Complex inputs not supported
    supports_complex = False

    def test_inverse_property(self):
        """Test dn(arcdn(x, k), k) = x."""
        from torchscience.special_functions import jacobi_elliptic_dn

        # x must be in range [sqrt(1-k^2), 1]
        k = torch.tensor([0.3, 0.4, 0.5])
        k_prime_sq = 1.0 - k ** 2
        x = torch.sqrt(k_prime_sq) + 0.1 * (1.0 - torch.sqrt(k_prime_sq))

        u = inverse_jacobi_elliptic_dn(x, k)
        result = jacobi_elliptic_dn(u, k)

        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_at_one(self):
        """Test arcdn(1, k) = 0."""
        x = torch.ones(5)
        k = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        expected = torch.zeros(5)
        torch.testing.assert_close(inverse_jacobi_elliptic_dn(x, k), expected, atol=1e-6, rtol=1e-5)
