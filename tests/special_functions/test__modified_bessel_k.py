import math

import torch
import scipy.special

from conftest import BinaryOperatorTestCase
from torchscience.special_functions import modified_bessel_k


class TestModifiedBesselK(BinaryOperatorTestCase):
    func = staticmethod(modified_bessel_k)
    op_name = "_modified_bessel_k"

    known_values = []  # K has singularity at x=0

    # Reference: scipy.special.kv
    reference = staticmethod(lambda nu, x: torch.from_numpy(
        scipy.special.kv(nu.numpy(), x.numpy())
    ).to(nu.dtype))

    input_range_1 = (0.0, 5.0)  # nu
    input_range_2 = (0.1, 5.0)  # x (positive only)

    gradcheck_inputs = ([0.0, 1.0, 2.0], [0.5, 1.0, 2.0])

    supports_complex = False

    def test_recurrence_relation(self):
        """Test K_{nu-1}(x) - K_{nu+1}(x) = -(2*nu/x) * K_nu(x)."""
        nu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        k_nu_minus_1 = modified_bessel_k(nu - 1, x)
        k_nu = modified_bessel_k(nu, x)
        k_nu_plus_1 = modified_bessel_k(nu + 1, x)

        lhs = k_nu_minus_1 - k_nu_plus_1
        rhs = -(2 * nu / x) * k_nu
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_specific_values(self):
        """Test specific values of K_nu(x)."""
        test_cases = [
            (0.0, 1.0, 0.4210244382),  # K_0(1)
            (1.0, 1.0, 0.6019072302),  # K_1(1)
            (0.0, 2.0, 0.1138938727),  # K_0(2)
            (1.0, 2.0, 0.1398658818),  # K_1(2)
        ]
        for nu_val, x_val, expected_val in test_cases:
            nu = torch.tensor([nu_val], dtype=torch.float64)
            x = torch.tensor([x_val], dtype=torch.float64)
            result = modified_bessel_k(nu, x)
            expected = torch.tensor([expected_val], dtype=torch.float64)
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_positive_for_positive_x(self):
        """Test K_nu(x) > 0 for x > 0."""
        nu = torch.linspace(0.0, 3.0, 10)
        x = torch.linspace(0.1, 5.0, 10)
        result = modified_bessel_k(nu, x)
        assert torch.all(result > 0), "K_nu(x) should be positive for x > 0"

    def test_symmetry_in_order(self):
        """Test K_{-nu}(x) = K_nu(x)."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        for nu_val in [0.5, 1.0, 1.5, 2.0]:
            nu_pos = torch.full_like(x, nu_val)
            nu_neg = torch.full_like(x, -nu_val)
            k_pos = modified_bessel_k(nu_pos, x)
            k_neg = modified_bessel_k(nu_neg, x)
            torch.testing.assert_close(k_pos, k_neg, atol=1e-5, rtol=1e-5)

    def test_half_integer_order(self):
        """Test K_{1/2}(x) = sqrt(pi/(2x)) * exp(-x)."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        nu = torch.full_like(x, 0.5)
        result = modified_bessel_k(nu, x)
        expected = torch.sqrt(math.pi / (2 * x)) * torch.exp(-x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_decreasing(self):
        """Test K_nu(x) is decreasing in x."""
        nu = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x1 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        x2 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)

        k1 = modified_bessel_k(nu, x1)
        k2 = modified_bessel_k(nu, x2)

        assert torch.all(k1 > k2), "K_nu should decrease with x"
