import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import inverse_jacobi_elliptic_sc


class TestInverseJacobiEllipticSc(BinaryOperatorTestCase):
    func = staticmethod(inverse_jacobi_elliptic_sc)
    op_name = "_inverse_jacobi_elliptic_sc"

    # Known values for arcsc(x, k)
    known_values = [
        ((0.0, 0.0), 0.0),  # arcsc(0, k) = 0
        ((0.0, 0.5), 0.0),  # arcsc(0, k) = 0
    ]

    # Input ranges
    input_range_1 = (-0.9, 0.9)  # x
    input_range_2 = (0.0, 0.9)   # k (modulus)

    # Gradcheck inputs
    gradcheck_inputs = ([0.1, 0.3, 0.5], [0.1, 0.3, 0.5])

    # Complex inputs not supported
    supports_complex = False

    def test_inverse_property(self):
        """Test sc(arcsc(x, k), k) = x."""
        from torchscience.special_functions import jacobi_elliptic_sc

        x = torch.tensor([0.1, 0.3, 0.5, 0.7])
        k = torch.tensor([0.2, 0.3, 0.4, 0.5])

        u = inverse_jacobi_elliptic_sc(x, k)
        result = jacobi_elliptic_sc(u, k)

        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_at_zero(self):
        """Test arcsc(0, k) = 0."""
        x = torch.zeros(5)
        k = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9])
        expected = torch.zeros(5)
        torch.testing.assert_close(inverse_jacobi_elliptic_sc(x, k), expected, atol=1e-6, rtol=1e-5)

    def test_odd_function(self):
        """Test arcsc(-x, k) = -arcsc(x, k)."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7])
        k = torch.tensor([0.2, 0.3, 0.4, 0.5])
        torch.testing.assert_close(
            inverse_jacobi_elliptic_sc(-x, k),
            -inverse_jacobi_elliptic_sc(x, k),
            atol=1e-5, rtol=1e-5
        )

    def test_with_k_zero(self):
        """Test arcsc(x, 0) = arctan(x) (trigonometric case)."""
        x = torch.tensor([0.0, 0.5, 1.0])
        k = torch.zeros_like(x)
        expected = torch.atan(x)
        torch.testing.assert_close(inverse_jacobi_elliptic_sc(x, k), expected, atol=1e-5, rtol=1e-5)
