"""Tests for chebyshev_polynomial_t using the reusable test framework.

The framework automatically tests:
- Autograd (gradcheck, gradgradcheck)
- Device handling (CPU, CUDA)
- Dtype handling (float32, float64, complex64, complex128)
- Low-precision dtypes (float16, bfloat16)
- Broadcasting
- torch.compile compatibility
- vmap support
- SymPy reference verification
- Recurrence relations
- Special values
"""

import math

import pytest
import sympy
import torch
import torch.testing
from sympy import I, N, symbols

import torchscience.special_functions
from torchscience.testing import (
    BinaryOpTestCase,
    InputSpec,
    OperatorDescriptor,
    RecurrenceSpec,
    SpecialValue,
    SymbolicDerivativeVerifier,
    ToleranceConfig,
)


def sympy_chebyshev_t(
    v: float | complex, z: float | complex
) -> float | complex:
    """Wrapper for SymPy Chebyshev T function via cos(v * acos(z))."""
    if isinstance(v, complex):
        sympy_v = sympy.Float(v.real) + I * sympy.Float(v.imag)
    else:
        sympy_v = sympy.Float(v)

    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)

    result = N(sympy.cos(sympy_v * sympy.acos(sympy_z)), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_chebyshev_t_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for Chebyshev polynomial T."""
    v, z = symbols("v z")
    # T_v(z) = cos(v * acos(z))
    expr = sympy.cos(v * sympy.acos(z))
    return SymbolicDerivativeVerifier(expr, [v, z])


def reference_chebyshev_t(v, z):
    """Reference implementation using torch.cos(v * torch.acos(z))."""
    if torch.is_complex(v) or torch.is_complex(z):
        if not torch.is_complex(z):
            z = z.to(
                torch.complex128
                if z.dtype == torch.float64
                else torch.complex64
            )
        if not torch.is_complex(v):
            v = v.to(z.dtype)
    return torch.cos(v * torch.acos(z))


def _check_recurrence(func) -> bool:
    """Check T_n(z) = 2z*T_{n-1}(z) - T_{n-2}(z)."""
    z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    for n in range(2, 10):
        v_n = torch.tensor([float(n)], dtype=torch.float64)
        v_n_1 = torch.tensor([float(n - 1)], dtype=torch.float64)
        v_n_2 = torch.tensor([float(n - 2)], dtype=torch.float64)

        left = func(v_n, z)
        right = 2 * z * func(v_n_1, z) - func(v_n_2, z)
        if not torch.allclose(left, right, rtol=1e-10, atol=1e-10):
            return False
    return True


class TestChebyshevPolynomialT(BinaryOpTestCase):
    """Tests for Chebyshev polynomial of the first kind."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="chebyshev_polynomial_t",
            func=torchscience.special_functions.chebyshev_polynomial_t,
            arity=2,
            input_specs=[
                InputSpec(
                    name="v",
                    position=0,
                    default_real_range=(0.0, 10.0),
                    can_be_integer=True,
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=1,
                    default_real_range=(-0.99, 0.99),
                    supports_grad=True,
                ),
            ],
            sympy_func=lambda v, z: sympy.cos(v * sympy.acos(z)),
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_gradgradcheck_complex",  # Complex 2nd order numerically sensitive
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="three_term_recurrence",
                    check_fn=_check_recurrence,
                    description="T_n(z) = 2z*T_{n-1}(z) - T_{n-2}(z)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=1.0,
                    description="T_0(z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.5),
                    expected=0.5,
                    description="T_1(z) = z",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=-0.5,
                    description="T_2(0.5) = 2*0.25-1",
                ),
            ],
            singularities=[],
            # Skip sparse/quantized/meta for binary operator
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=False,
        )

    # =========================================================================
    # Chebyshev-specific tests (not covered by framework)
    # =========================================================================

    def test_integer_degree_polynomial_values(self):
        """Test exact polynomial values for integer degrees."""
        z = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], dtype=torch.float64)

        # T_0(x) = 1
        v = torch.tensor([0.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected)

        # T_1(x) = x
        v = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        torch.testing.assert_close(result, z)

        # T_2(x) = 2x^2 - 1
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = 2 * z**2 - 1
        torch.testing.assert_close(result, expected)

        # T_3(x) = 4x^3 - 3x
        v = torch.tensor([3.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = 4 * z**3 - 3 * z
        torch.testing.assert_close(result, expected)

    def test_negative_degree_symmetry(self):
        """Test T_{-n}(z) = T_n(z)."""
        z = torch.tensor([0.5, 0.3, -0.7], dtype=torch.float64)
        for n in range(6):
            v_pos = torch.tensor([float(n)], dtype=torch.float64)
            v_neg = torch.tensor([float(-n)], dtype=torch.float64)
            result_pos = torchscience.special_functions.chebyshev_polynomial_t(
                v_pos, z
            )
            result_neg = torchscience.special_functions.chebyshev_polynomial_t(
                v_neg, z
            )
            torch.testing.assert_close(
                result_pos, result_neg, rtol=1e-10, atol=1e-10
            )

    def test_non_integer_degree(self):
        """Test non-integer degree uses analytic continuation."""
        z = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_z_real_v(self):
        """Test complex z with real v."""
        z = torch.tensor(
            [1.0 + 0.1j, 0.5 + 0.5j, -0.5 - 0.5j], dtype=torch.complex128
        )
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_z_complex_v(self):
        """Test complex z with complex v."""
        z = torch.tensor([0.5 + 0.2j], dtype=torch.complex128)
        v = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_continuation_z_greater_than_1(self):
        """Test T_v(z) = cosh(v * acosh(z)) for z > 1."""
        z = torch.tensor([1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.cosh(v * torch.acosh(z))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_hyperbolic_continuation_z_less_than_minus_1(self):
        """Test T_v(z) = cos(v*pi) * cosh(v * acosh(-z)) for z < -1."""
        z = torch.tensor([-1.5, -2.0, -3.0, -5.0], dtype=torch.float64)
        v = torch.tensor([0.25], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        expected = torch.cos(v * torch.tensor(math.pi)) * torch.cosh(
            v * torch.acosh(-z)
        )
        torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_z_outside_domain_integer_v(self):
        """Test that real z outside [-1, 1] with integer v uses recurrence."""
        z = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)
        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        # T_2(z) = 2z^2 - 1
        expected = torch.tensor([3.5, 7.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_continuity_at_z_equals_1(self):
        """Test continuity at z = 1."""
        v = torch.tensor([2.5], dtype=torch.float64)
        z_inside = torch.tensor([1.0 - 1e-8], dtype=torch.float64)
        z_outside = torch.tensor([1.0 + 1e-8], dtype=torch.float64)

        result_inside = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_inside
        )
        result_outside = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_outside
        )

        # Both should be close to T_v(1) = 1
        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            result_inside, expected, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            result_outside, expected, rtol=1e-6, atol=1e-6
        )

    def test_branch_cut_near_plus_one(self):
        """Test behavior near z = 1 with small imaginary parts."""
        z_above = torch.tensor([1.0 + 0.01j], dtype=torch.complex128)
        z_below = torch.tensor([1.0 - 0.01j], dtype=torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)

        result_above = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_above
        )
        result_below = torchscience.special_functions.chebyshev_polynomial_t(
            v, z_below
        )

        # Results should be conjugates
        torch.testing.assert_close(
            result_above, result_below.conj(), rtol=1e-6, atol=1e-6
        )

    def test_gradcheck_z_greater_than_1(self):
        """Test gradient correctness for z > 1."""
        z = torch.tensor(
            [1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_z_less_than_minus_1(self):
        """Test gradient correctness for z < -1."""
        z = torch.tensor(
            [-1.5, -2.0, -3.0], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_v_noninteger(self):
        """Test gradient correctness for non-integer v."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return torchscience.special_functions.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(
            func, (v,), eps=1e-6, atol=1e-4, rtol=1e-4
        )

    def test_boundary_gradient_at_z_equals_1(self):
        """Test gradient at z = 1 boundary."""
        z = torch.tensor(
            [1.0 - 1e-10], dtype=torch.float64, requires_grad=True
        )
        v = torch.tensor([2.5], dtype=torch.float64)

        y = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        y.backward()

        # At z = 1: dT_v/dz = v^2
        expected_grad = v * v
        torch.testing.assert_close(z.grad, expected_grad, rtol=1e-3, atol=1e-3)

    def test_unrolling_boundary(self):
        """Test degrees at the unrolling boundary (n=7,8,9) work correctly."""
        z = torch.linspace(-1, 1, 50, dtype=torch.float64)
        for n in [6, 7, 8, 9, 10, 15, 20]:
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype):
        """Test forward pass with low-precision dtypes."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)

        result = torchscience.special_functions.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype

        # Compare against float32 reference
        z_f32 = z.to(torch.float32)
        v_f32 = v.to(torch.float32)
        expected = torchscience.special_functions.chebyshev_polynomial_t(
            v_f32, z_f32
        )

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    def test_integer_degree_matches_reference(self):
        """Test that integer degree recurrence matches cos(n*arccos(x))."""
        z = torch.linspace(-1, 1, 100, dtype=torch.float64)
        for n in range(10):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = torchscience.special_functions.chebyshev_polynomial_t(
                v, z
            )
            expected = reference_chebyshev_t(v, z)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )
