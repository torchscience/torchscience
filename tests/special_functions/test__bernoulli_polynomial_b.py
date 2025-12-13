import torch

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import bernoulli_polynomial_b


class TestBernoulliPolynomialB(BinaryOperatorTestCase):
    func = staticmethod(bernoulli_polynomial_b)
    op_name = "_bernoulli_polynomial_b"

    # Known values for Bernoulli polynomials
    # B_0(x) = 1
    # B_1(x) = x - 1/2
    # B_2(x) = x^2 - x + 1/6
    known_values = [
        ((0.0, 0.0), 1.0),       # B_0(0) = 1
        ((0.0, 1.0), 1.0),       # B_0(1) = 1
        ((1.0, 0.0), -0.5),      # B_1(0) = -1/2
        ((1.0, 0.5), 0.0),       # B_1(0.5) = 0
        ((1.0, 1.0), 0.5),       # B_1(1) = 1/2
        ((2.0, 0.0), 1.0/6.0),   # B_2(0) = 1/6
        ((2.0, 1.0), 1.0/6.0),   # B_2(1) = 1/6
    ]

    # No standard reference implementation
    reference = None

    # Input ranges
    input_range_1 = (0.0, 5.0)  # n must be non-negative integer
    input_range_2 = (-2.0, 2.0)  # x can be any real

    # Gradcheck inputs (using integer n values)
    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.25, 0.5, 0.75])

    # Bernoulli polynomials support real inputs but n should be integer
    supports_complex = False

    def test_b0_constant(self):
        """Test B_0(x) = 1 for all x."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = torch.zeros_like(x)
        output = bernoulli_polynomial_b(n, x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_b1_linear(self):
        """Test B_1(x) = x - 1/2."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = torch.ones_like(x)
        output = bernoulli_polynomial_b(n, x)
        expected = x - 0.5
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_b2_quadratic(self):
        """Test B_2(x) = x^2 - x + 1/6."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        n = 2.0 * torch.ones_like(x)
        output = bernoulli_polynomial_b(n, x)
        expected = x**2 - x + 1.0/6.0
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_symmetry_at_half(self):
        """Test B_n(1-x) = (-1)^n B_n(x) for integer n."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        for n_val in [0, 1, 2, 3, 4]:
            n = float(n_val) * torch.ones_like(x)
            lhs = bernoulli_polynomial_b(n, 1.0 - x)
            rhs = ((-1.0) ** n_val) * bernoulli_polynomial_b(n, x)
            torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_derivative_relation(self):
        """Test d/dx B_n(x) = n * B_{n-1}(x) using finite differences."""
        x = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True)
        n = 3.0 * torch.ones_like(x)

        # Compute gradient via autograd
        output = bernoulli_polynomial_b(n, x)
        grad_x = torch.autograd.grad(output.sum(), x)[0]

        # Expected: n * B_{n-1}(x)
        with torch.no_grad():
            expected = 3.0 * bernoulli_polynomial_b(2.0 * torch.ones_like(x), x)

        torch.testing.assert_close(grad_x, expected, atol=1e-5, rtol=1e-5)
