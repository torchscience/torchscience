import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import euler_polynomial_e


class TestEulerPolynomialE(BinaryOperatorTestCase):
    func = staticmethod(euler_polynomial_e)
    op_name = "_euler_polynomial_e"

    # Known values for Euler polynomials
    # E_0(x) = 1
    # E_1(x) = x - 1/2
    # E_2(x) = x^2 - x
    known_values = [
        ((0.0, 0.0), 1.0),       # E_0(0) = 1
        ((0.0, 1.0), 1.0),       # E_0(1) = 1
        ((1.0, 0.0), -0.5),      # E_1(0) = -1/2
        ((1.0, 0.5), 0.0),       # E_1(0.5) = 0
        ((1.0, 1.0), 0.5),       # E_1(1) = 1/2
        ((2.0, 0.0), 0.0),       # E_2(0) = 0
        ((2.0, 1.0), 0.0),       # E_2(1) = 0
        ((2.0, 0.5), -0.25),     # E_2(0.5) = -1/4
    ]

    # No standard reference implementation
    reference = None

    # Input ranges
    input_range_1 = (0.0, 5.0)  # n must be non-negative integer
    input_range_2 = (-2.0, 2.0)  # x can be any real

    # Gradcheck inputs
    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.25, 0.5, 0.75])

    # Euler polynomials support real inputs
    supports_complex = False

    def test_e0_constant(self):
        """Test E_0(x) = 1 for all x."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = torch.zeros_like(x)
        output = euler_polynomial_e(n, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_e1_linear(self):
        """Test E_1(x) = x - 1/2."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = torch.ones_like(x)
        output = euler_polynomial_e(n, x)
        expected = x - 0.5
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_e2_quadratic(self):
        """Test E_2(x) = x^2 - x."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = 2.0 * torch.ones_like(x)
        output = euler_polynomial_e(n, x)
        expected = x**2 - x
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_symmetry(self):
        """Test E_n(1-x) = (-1)^n E_n(x)."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        for n_val in [0, 1, 2, 3]:
            n = float(n_val) * torch.ones_like(x)
            lhs = euler_polynomial_e(n, 1.0 - x)
            rhs = ((-1.0) ** n_val) * euler_polynomial_e(n, x)
            torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_derivative_relation(self):
        """Test d/dx E_n(x) = n * E_{n-1}(x) using autograd."""
        x = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True)
        n = 3.0 * torch.ones_like(x)

        # Compute gradient via autograd
        output = euler_polynomial_e(n, x)
        grad_x = torch.autograd.grad(output.sum(), x)[0]

        # Expected: n * E_{n-1}(x)
        with torch.no_grad():
            expected = 3.0 * euler_polynomial_e(2.0 * torch.ones_like(x), x)

        torch.testing.assert_close(grad_x, expected, atol=1e-5, rtol=1e-5)
