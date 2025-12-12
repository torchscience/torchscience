import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import sinc_pi


class TestSincPi(UnaryOperatorTestCase):
    func = staticmethod(sinc_pi)
    op_name = "_sinc_pi"

    # sinc_pi is an even function
    symmetry = "even"
    period = None

    # sinc_pi is bounded between approximately -0.217 and 1
    bounds = (-0.25, 1.0)

    known_values = {
        0.0: 1.0,  # sinc_pi(0) = 1
        0.5: 2.0 / math.pi,  # sinc_pi(0.5) = sin(pi/2)/(pi/2) = 2/pi
        1.0: 0.0,  # sinc_pi(1) = 0
        2.0: 0.0,  # sinc_pi(2) = 0
        3.0: 0.0,  # sinc_pi(3) = 0
    }

    # sinc_pi(n) = 0 for nonzero integers
    zeros = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]

    # Reference: compute manually as sin(pi*x)/(pi*x)
    reference = None

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-10.0, 10.0)

    gradcheck_inputs = [0.1, 0.25, 0.5, 1.5, 2.5]

    preserves_negative_zero = True

    extreme_values = [1e-10, 1e-5, 0.001, 10.0, 100.0]

    def test_at_zero(self):
        """Test sinc_pi(0) = 1."""
        result = sinc_pi(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([1.0]), atol=1e-10, rtol=0)

    def test_zeros_at_integers(self):
        """Test sinc_pi(n) = 0 for nonzero integers."""
        integers = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = sinc_pi(integers)
        torch.testing.assert_close(
            result, torch.zeros_like(integers), atol=1e-10, rtol=0
        )

    def test_symmetry_even(self):
        """Test sinc_pi(-x) = sinc_pi(x) (even function)."""
        x = torch.tensor([0.1, 0.5, 1.5, 2.5, 3.5])
        torch.testing.assert_close(sinc_pi(-x), sinc_pi(x), atol=1e-7, rtol=1e-6)

    def test_half_integer_values(self):
        """Test sinc_pi at half-integer values."""
        # sinc_pi(n + 0.5) = sin(pi*(n+0.5)) / (pi*(n+0.5))
        # = sin(n*pi + pi/2) / (pi*(n+0.5))
        # = cos(n*pi) / (pi*(n+0.5))
        # = (-1)^n / (pi*(n+0.5))
        x = torch.tensor([0.5, 1.5, 2.5, 3.5])
        expected = torch.tensor(
            [
                2.0 / math.pi,  # 1 / (pi * 0.5)
                -2.0 / (3 * math.pi),  # -1 / (pi * 1.5)
                2.0 / (5 * math.pi),  # 1 / (pi * 2.5)
                -2.0 / (7 * math.pi),  # -1 / (pi * 3.5)
            ]
        )
        torch.testing.assert_close(sinc_pi(x), expected, atol=1e-6, rtol=1e-6)

    def test_derivative_at_zero(self):
        """Test d/dx sinc_pi(x)|_{x=0} = 0."""
        x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        y = sinc_pi(x)
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.0], dtype=torch.float64), atol=1e-10, rtol=0)

    def test_derivative(self):
        """Test d/dx sinc_pi(x) = (cos_pi(x) - sinc_pi(x)) / x."""
        from torchscience.special_functions import cos_pi

        x = torch.tensor([0.25, 0.5, 1.5, 2.5], dtype=torch.float64, requires_grad=True)
        y = sinc_pi(x)
        y.sum().backward()

        x_detached = x.detach()
        expected_grad = (cos_pi(x_detached) - sinc_pi(x_detached)) / x_detached
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_relation_to_sin_pi(self):
        """Test sinc_pi(x) = sin_pi(x) / (pi * x) for x != 0."""
        from torchscience.special_functions import sin_pi

        x = torch.tensor([0.1, 0.25, 0.5, 1.5, 2.5])
        expected = sin_pi(x) / (math.pi * x)
        torch.testing.assert_close(sinc_pi(x), expected, atol=1e-7, rtol=1e-6)

    def test_bounded(self):
        """Test |sinc_pi(x)| <= 1."""
        x = torch.linspace(-20.0, 20.0, 1000)
        result = sinc_pi(x)
        assert torch.all(result.abs() <= 1.0 + 1e-6), "sinc_pi should be bounded by 1"

    def test_decay(self):
        """Test sinc_pi decays as 1/x for large |x|."""
        x = torch.tensor([10.0, 20.0, 50.0, 100.0])
        result = sinc_pi(x).abs()
        bound = 1.0 / (math.pi * x.abs())
        assert torch.all(result <= bound + 1e-6), "sinc_pi should decay as 1/(pi*x)"

    def test_small_x_taylor(self):
        """Test sinc_pi(x) ≈ 1 - (pi*x)^2/6 for small x."""
        x = torch.tensor([1e-4, 1e-3, 1e-2])
        taylor_approx = 1 - (math.pi * x) ** 2 / 6
        torch.testing.assert_close(sinc_pi(x), taylor_approx, atol=1e-6, rtol=1e-4)

    def test_numerical_stability_near_zero(self):
        """Test numerical stability for very small x."""
        small_x = torch.tensor([1e-15, 1e-10, 1e-8, 1e-6])
        result = sinc_pi(small_x)
        # Should be very close to 1 for small x
        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-5, rtol=1e-5)

    def test_compare_with_torch_sinc(self):
        """Test relation to torch.special.sinc (unnormalized)."""
        # torch.special.sinc(x) = sin(pi*x) / (pi*x) = sinc_pi(x)
        # Note: torch.special.sinc is already the normalized sinc
        x = torch.linspace(-5.0, 5.0, 50)
        # Avoid exact zero for torch.special.sinc comparison
        x = x[x != 0]
        result = sinc_pi(x)
        expected = torch.special.sinc(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_complex_input(self):
        """Test sinc_pi with complex input."""
        z = torch.tensor([0.5 + 0.5j, 1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
        result = sinc_pi(z)
        # sinc_pi should return complex values
        assert result.dtype == torch.complex64
