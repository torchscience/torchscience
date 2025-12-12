import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import inverse_jacobi_elliptic_cd


class TestInverseJacobiEllipticCd(BinaryOperatorTestCase):
    func = staticmethod(inverse_jacobi_elliptic_cd)
    op_name = "_inverse_jacobi_elliptic_cd"

    # Known values for arccd(x, k)
    known_values = [
        ((1.0, 0.0), 0.0),  # arccd(1, k) = 0
        ((1.0, 0.5), 0.0),  # arccd(1, k) = 0
    ]

    # Input ranges: x in (-1, 1), k in [0, 1)
    input_range_1 = (0.1, 0.99)  # x
    input_range_2 = (0.0, 0.9)   # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.3, 0.5, 0.7], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_inverse_property(self):
        """Test cd(arccd(x, k), k) = x."""
        from torchscience.special_functions import jacobi_elliptic_cd

        x = torch.tensor([0.3, 0.5, 0.7, 0.9])
        k = torch.tensor([0.2, 0.3, 0.4, 0.5])

        u = inverse_jacobi_elliptic_cd(x, k)
        result = jacobi_elliptic_cd(u, k)

        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_at_one(self):
        """Test arccd(1, k) = 0."""
        x = torch.ones(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9])
        expected = torch.zeros(5)
        torch.testing.assert_close(inverse_jacobi_elliptic_cd(x, k), expected, atol=1e-6, rtol=1e-5)
