import math

import pytest
import torch

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestAssociatedLegendrePolynomialP(OpTestCase):
    """Tests for the associated_legendre_polynomial_p function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="associated_legendre_polynomial_p",
            func=torchscience.special_functions.associated_legendre_polynomial_p,
            arity=3,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.0, 6.0)),
                InputSpec(name="m", position=1, default_real_range=(0.0, 3.0)),
                InputSpec(
                    name="x",
                    position=2,
                    default_real_range=(-0.95, 0.95),
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
                "test_nan_propagation_all_inputs",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0, 0.5),
                    expected=1.0,
                    description="P_0^0(x) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 0.5),
                    expected=0.5,
                    description="P_1^0(x) = x",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 0.5),
                    expected=-0.125,
                    description="P_2^0(0.5) = (3*0.25 - 1)/2",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0, 0.5),
                    expected=2.25,
                    description="P_2^2(0.5) = 3*(1 - 0.25)",
                ),
                SpecialValue(
                    inputs=(3.0, 0.0, 0.5),
                    expected=-0.4375,
                    description="P_3^0(0.5) = (5*0.125 - 1.5)/2",
                ),
                SpecialValue(
                    inputs=(0.0, 0.0, 1.0),
                    expected=1.0,
                    description="P_0^0(1) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 0.0, -1.0),
                    expected=1.0,
                    description="P_0^0(-1) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )


class TestAssociatedLegendrePolynomialPCustom:
    """Custom tests beyond the framework."""

    def test_p11_formula(self):
        """P_1^1(x) = -sqrt(1 - x^2)."""
        x_vals = [0.0, 0.3, 0.5, 0.8]
        n = torch.tensor([1.0])
        m = torch.tensor([1.0])
        for xv in x_vals:
            x = torch.tensor([xv])
            result = torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
            expected = -math.sqrt(1 - xv * xv)
            torch.testing.assert_close(
                result,
                torch.tensor([expected]),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_p21_formula(self):
        """P_2^1(x) = -3x * sqrt(1 - x^2)."""
        x_vals = [0.0, 0.3, 0.5, 0.8]
        n = torch.tensor([2.0])
        m = torch.tensor([1.0])
        for xv in x_vals:
            x = torch.tensor([xv])
            result = torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
            expected = -3 * xv * math.sqrt(1 - xv * xv)
            torch.testing.assert_close(
                result,
                torch.tensor([expected]),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_vs_scipy(self):
        """Compare with scipy.special.lpmv for various (n, m) pairs."""
        try:
            from scipy.special import lpmv
        except ImportError:
            pytest.skip("scipy not available")

        x = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)
        test_cases = [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, 4),
            (5, 0),
            (5, 3),
            (6, 0),
            (6, 4),
        ]
        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m = torch.tensor([float(m_val)], dtype=torch.float64)
            result = torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
            # scipy.special.lpmv(m, n, x) — note: scipy uses (m, n, x) order
            expected = torch.tensor(
                lpmv(m_val, n_val, x.numpy()), dtype=torch.float64
            )
            torch.testing.assert_close(
                result,
                expected,
                atol=1e-10,
                rtol=1e-10,
                msg=f"Failed for n={n_val}, m={m_val}",
            )

    def test_negative_m_symmetry(self):
        """P_n^{-m} = (-1)^m * (n-m)!/(n+m)! * P_n^m."""
        x = torch.tensor([0.3, 0.7], dtype=torch.float64)
        test_cases = [(2, 1), (3, 2), (4, 3), (5, 2)]
        for n_val, m_val in test_cases:
            n = torch.tensor([float(n_val)], dtype=torch.float64)
            m_pos = torch.tensor([float(m_val)], dtype=torch.float64)
            m_neg = torch.tensor([float(-m_val)], dtype=torch.float64)

            P_pos = torchscience.special_functions.associated_legendre_polynomial_p(
                n, m_pos, x
            )
            P_neg = torchscience.special_functions.associated_legendre_polynomial_p(
                n, m_neg, x
            )

            sign = (-1) ** m_val
            factor = math.factorial(n_val - m_val) / math.factorial(
                n_val + m_val
            )
            expected = sign * factor * P_pos

            torch.testing.assert_close(
                P_neg,
                expected,
                atol=1e-10,
                rtol=1e-10,
                msg=f"Symmetry failed for n={n_val}, m={m_val}",
            )

    def test_invalid_m_greater_than_n(self):
        """P_n^m = 0 when |m| > n."""
        n = torch.tensor([2.0])
        m = torch.tensor([3.0])
        x = torch.tensor([0.5])
        result = (
            torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
        )
        torch.testing.assert_close(result, torch.tensor([0.0]))

    def test_broadcasting(self):
        """Test that n, m, x broadcast correctly."""
        # n shape (3,), m shape (1,), x shape (4,) -> output shape (3, 4)
        n = torch.tensor([0.0, 1.0, 2.0]).unsqueeze(1)  # (3, 1)
        m = torch.tensor([0.0])  # (1,)
        x = torch.tensor([0.0, 0.25, 0.5, 0.75])  # (4,)
        result = (
            torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
        )
        assert result.shape == (3, 4)
        # Verify P_0^0(x) = 1
        torch.testing.assert_close(result[0], torch.ones(4))
        # Verify P_1^0(x) = x
        torch.testing.assert_close(result[1], x)

    def test_gradcheck_x(self):
        """Gradient check for x (continuous parameter)."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(x_in):
            return torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x_in
            )

        torch.autograd.gradcheck(func, (x,), raise_exception=True)

    def test_grad_n_m_zero(self):
        """Gradients w.r.t. n and m should be zero."""
        n = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
        m = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        result = (
            torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
        )
        result.backward()

        torch.testing.assert_close(
            n.grad, torch.tensor([0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            m.grad, torch.tensor([0.0], dtype=torch.float64)
        )

    def test_gradgradcheck_x(self):
        """Second-order gradient check for x."""
        n = torch.tensor([3.0], dtype=torch.float64)
        m = torch.tensor([1.0], dtype=torch.float64)
        x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        def func(x_in):
            return torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x_in
            )

        torch.autograd.gradgradcheck(
            func, (x,), atol=1e-3, raise_exception=True
        )

    def test_meta_tensor(self):
        """Meta tensor shape inference."""
        n = torch.tensor([2.0], device="meta")
        m = torch.tensor([1.0], device="meta")
        x = torch.tensor([0.5], device="meta")
        result = (
            torchscience.special_functions.associated_legendre_polynomial_p(
                n, m, x
            )
        )
        assert result.shape == (1,)
        assert result.device.type == "meta"
