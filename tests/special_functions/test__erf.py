import math

import torch

from conftest import UnaryOperatorTestCase
from torchscience.special_functions import erf


class TestErf(UnaryOperatorTestCase):
    func = staticmethod(erf)
    op_name = "_erf"

    # erf is an odd function
    symmetry = "odd"
    period = None

    # erf is bounded between -1 and 1
    bounds = (-1.0, 1.0)

    known_values = {
        0.0: 0.0,
        1.0: 0.8427007929497149,  # erf(1)
        2.0: 0.9953222650189527,  # erf(2)
        3.0: 0.9999779095030014,  # erf(3)
    }

    zeros = [0.0]

    # Reference: use torch.special.erf (PyTorch's built-in)
    reference = staticmethod(lambda x: torch.special.erf(x))

    reference_atol = 1e-6
    reference_rtol = 1e-5

    input_range = (-5.0, 5.0)

    gradcheck_inputs = [0.1, 0.5, 1.0, 2.0]

    preserves_negative_zero = True

    extreme_values = [1e-30, 1e-10, 1e-5, 5.0, 10.0]

    def test_symmetry_odd(self):
        """Test erf(-x) = -erf(x)."""
        x = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
        torch.testing.assert_close(erf(-x), -erf(x), atol=1e-7, rtol=1e-6)

    def test_limits(self):
        """Test erf approaches ±1 for large |x|."""
        large_pos = torch.tensor([5.0, 10.0, 20.0])
        large_neg = torch.tensor([-5.0, -10.0, -20.0])

        # erf(x) -> 1 as x -> inf
        torch.testing.assert_close(
            erf(large_pos), torch.ones_like(large_pos), atol=1e-6, rtol=1e-6
        )

        # erf(x) -> -1 as x -> -inf
        torch.testing.assert_close(
            erf(large_neg), -torch.ones_like(large_neg), atol=1e-6, rtol=1e-6
        )

    def test_derivative(self):
        """Test d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
        y = erf(x)
        y.sum().backward()

        two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
        expected_grad = two_over_sqrt_pi * torch.exp(-x.detach() ** 2)
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-5, rtol=1e-5)

    def test_relation_to_erfc(self):
        """Test erf(x) + erfc(x) = 1."""
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
        # erfc(x) = 1 - erf(x)
        erfc_values = 1 - erf(x)
        torch.testing.assert_close(
            erf(x) + erfc_values, torch.ones_like(x), atol=1e-7, rtol=1e-6
        )

    def test_small_x_approximation(self):
        """Test erf(x) ≈ (2/sqrt(pi)) * x for small x."""
        x = torch.tensor([1e-10, 1e-8, 1e-6, 1e-4])
        two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
        approx = two_over_sqrt_pi * x
        torch.testing.assert_close(erf(x), approx, atol=1e-10, rtol=1e-4)

    def test_monotonically_increasing(self):
        """Test erf is monotonically increasing."""
        x = torch.linspace(-5.0, 5.0, 100)
        output = erf(x)
        diff = output[1:] - output[:-1]
        assert torch.all(diff >= 0), "erf should be monotonically increasing"

    def test_special_values(self):
        """Test erf at special values."""
        # erf(0) = 0
        torch.testing.assert_close(
            erf(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-10, rtol=0
        )

        # erf(inf) = 1
        torch.testing.assert_close(
            erf(torch.tensor([float("inf")])),
            torch.tensor([1.0]),
            atol=1e-10,
            rtol=0,
        )

        # erf(-inf) = -1
        torch.testing.assert_close(
            erf(torch.tensor([float("-inf")])),
            torch.tensor([-1.0]),
            atol=1e-10,
            rtol=0,
        )

    def test_maclaurin_series(self):
        """Test erf against Maclaurin series for small x."""
        # erf(x) = (2/sqrt(pi)) * (x - x^3/3 + x^5/10 - x^7/42 + ...)
        x = torch.tensor([0.1, 0.2, 0.3])
        two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
        series = two_over_sqrt_pi * (
            x - x**3 / 3 + x**5 / 10 - x**7 / 42 + x**9 / 216
        )
        torch.testing.assert_close(erf(x), series, atol=1e-6, rtol=1e-5)

    def test_relation_to_normal_cdf(self):
        """Test relation to standard normal CDF: Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))."""
        x = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
        # Standard normal CDF
        phi = 0.5 * (1 + erf(x / math.sqrt(2)))
        # Compare with torch's implementation
        expected = torch.distributions.Normal(0, 1).cdf(x)
        torch.testing.assert_close(phi, expected, atol=1e-5, rtol=1e-5)
