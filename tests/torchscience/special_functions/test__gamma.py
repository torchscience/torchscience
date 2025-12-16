"""Tests for gamma function using the reusable test framework.

The framework automatically tests:
- Autograd (gradcheck, gradgradcheck)
- Device handling (CPU, CUDA)
- Dtype handling (float32, float64, complex64, complex128)
- Low-precision dtypes (float16, bfloat16)
- Broadcasting
- torch.compile compatibility
- vmap support
- Sparse tensor support (COO, CSR)
- Quantized tensor support
- Meta tensor support
- Autocast (mixed precision)
- NaN/Inf propagation
- SymPy reference verification
- Recurrence relations
- Functional identities
- Special values
- Singularity behavior
"""

import math

import pytest
import sympy
import torch
import torch.testing
from sympy import I, N, symbols

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    RecurrenceSpec,
    SingularitySpec,
    SpecialValue,
    SymbolicDerivativeVerifier,
    ToleranceConfig,
)


def sympy_gamma(z: float | complex) -> float | complex:
    """Wrapper for SymPy gamma function."""
    if isinstance(z, complex):
        sympy_z = sympy.Float(z.real) + I * sympy.Float(z.imag)
    else:
        sympy_z = sympy.Float(z)
    result = N(sympy.gamma(sympy_z), 50)
    if result.is_real:
        return float(result)
    return complex(result)


def create_gamma_verifier() -> SymbolicDerivativeVerifier:
    """Create derivative verifier for the gamma function."""
    z = symbols("z")
    expr = sympy.gamma(z)
    return SymbolicDerivativeVerifier(expr, [z])


def _check_recurrence(func) -> bool:
    """Check Gamma(x+1) = x * Gamma(x)."""
    x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
    left = func(x + 1)
    right = x * func(x)
    return torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def _reflection_identity(func):
    """Check Gamma(x) * Gamma(1-x) = pi / sin(pi*x)."""
    x = torch.tensor([0.25, 0.3, 0.4, 0.6], dtype=torch.float64)
    left = func(x) * func(1 - x)
    right = math.pi / torch.sin(math.pi * x)
    return left, right


def _duplication_identity(func):
    """Check Legendre duplication formula."""
    z = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    left = func(z) * func(z + 0.5)
    right = math.sqrt(math.pi) / (2 ** (2 * z - 1)) * func(2 * z)
    return left, right


class TestGamma(OpTestCase):
    """Tests for the gamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        sqrt_pi = math.sqrt(math.pi)
        return OperatorDescriptor(
            name="gamma",
            func=torchscience.special_functions.gamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(0.5, 20.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            sympy_func=sympy.gamma,
            tolerances=ToleranceConfig(),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                "test_gradgradcheck_complex",  # Complex 2nd order numerically sensitive
                "test_sparse_coo_basic",  # Sparse has implicit zeros = poles
                "test_low_precision_forward",  # Random values may hit poles
            },
            recurrence_relations=[
                RecurrenceSpec(
                    name="gamma_recurrence",
                    check_fn=_check_recurrence,
                    description="Gamma(x+1) = x * Gamma(x)",
                ),
            ],
            functional_identities=[
                IdentitySpec(
                    name="reflection_formula",
                    identity_fn=_reflection_identity,
                    description="Gamma(x) * Gamma(1-x) = pi / sin(pi*x)",
                ),
                IdentitySpec(
                    name="duplication_formula",
                    identity_fn=_duplication_identity,
                    description="Legendre duplication formula",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=1.0,
                    description="Gamma(1) = 0! = 1",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1.0,
                    description="Gamma(2) = 1! = 1",
                ),
                SpecialValue(
                    inputs=(3.0,),
                    expected=2.0,
                    description="Gamma(3) = 2! = 2",
                ),
                SpecialValue(
                    inputs=(4.0,),
                    expected=6.0,
                    description="Gamma(4) = 3! = 6",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=24.0,
                    description="Gamma(5) = 4! = 24",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=sqrt_pi,
                    description="Gamma(1/2) = sqrt(pi)",
                ),
                SpecialValue(
                    inputs=(1.5,),
                    expected=sqrt_pi / 2,
                    description="Gamma(3/2) = sqrt(pi)/2",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: (float(n) for n in range(-100, 1)),
                    expected_behavior="inf",
                    description="Poles at non-positive integers",
                ),
            ],
            supports_sparse_coo=True,
            supports_sparse_csr=True,
            supports_quantized=True,
            supports_meta=True,
        )

    # =========================================================================
    # Gamma-specific tests (not covered by framework)
    # =========================================================================

    def test_factorial_relationship(self):
        """Test Gamma(n) = (n-1)! for positive integers."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        expected = torch.tensor(
            [1.0, 1.0, 2.0, 6.0, 24.0, 120.0], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_half_integer_values(self):
        """Test Gamma at half-integer values."""
        sqrt_pi = math.sqrt(math.pi)
        x = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        expected = torch.tensor(
            [sqrt_pi, sqrt_pi / 2, 3 * sqrt_pi / 4], dtype=torch.float64
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_non_integer_values(self):
        """Test Gamma at negative non-integer values."""
        x = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        expected = torch.tensor(
            [
                -2 * math.sqrt(math.pi),
                4 * math.sqrt(math.pi) / 3,
                -8 * math.sqrt(math.pi) / 15,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_poles_return_inf(self):
        """Test that Gamma at poles returns inf."""
        poles = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64)
        result = torchscience.special_functions.gamma(poles)
        assert (torch.isinf(result) | torch.isnan(result)).all()

    def test_complex_conjugate_symmetry(self):
        """Test Gamma(conj(z)) = conj(Gamma(z))."""
        z = torch.tensor(
            [1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128
        )
        result_z = torchscience.special_functions.gamma(z)
        result_conj_z = torchscience.special_functions.gamma(z.conj())
        torch.testing.assert_close(
            result_conj_z, result_z.conj(), rtol=1e-10, atol=1e-10
        )

    def test_complex_positive_real_axis_matches_real(self):
        """Test complex numbers on positive real axis match real gamma."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)
        result_real = torchscience.special_functions.gamma(x_real)
        result_complex = torchscience.special_functions.gamma(x_complex)
        torch.testing.assert_close(
            result_complex.real, result_real, rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            result_complex.imag,
            torch.zeros_like(result_real),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_backward_formula_manual(self):
        """Manually verify backward formula: d/dx Gamma(x) = Gamma(x) * psi(x)."""
        x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        y = torchscience.special_functions.gamma(x)
        y.backward()
        gamma_euler = 0.5772156649015329
        expected_grad = 1.0 * (1.0 - gamma_euler)
        torch.testing.assert_close(
            x.grad,
            torch.tensor([expected_grad], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_large_positive_values(self):
        """Test with large positive values."""
        x = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result[0],
            torch.tensor(362880.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_small_positive_values_approximation(self):
        """Test Gamma(x) ~ 1/x for small x > 0."""
        x = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        expected_approx = 1 / x
        assert torch.isfinite(result).all()
        rel_error = torch.abs(result - expected_approx) / torch.abs(
            expected_approx
        )
        assert (rel_error < 0.1).all()

    def test_lut_boundary_float32(self):
        """Test LUT boundary for float32 (Gamma(35) should work, Gamma(36) overflows)."""
        x = torch.tensor([35.0], dtype=torch.float32)
        result = torchscience.special_functions.gamma(x)
        assert torch.isfinite(result).all()

    def test_lut_boundary_float64(self):
        """Test LUT boundary for float64 (Gamma(171) should work, Gamma(172) overflows)."""
        x = torch.tensor([171.0], dtype=torch.float64)
        result = torchscience.special_functions.gamma(x)
        assert torch.isfinite(result).all()

        x_overflow = torch.tensor([172.0], dtype=torch.float64)
        result_overflow = torchscience.special_functions.gamma(x_overflow)
        assert torch.isinf(result_overflow).all()

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_dtype_requires_conversion(self, int_dtype):
        """Test that integer inputs require explicit conversion to float."""
        x_int = torch.tensor([1, 2, 3], dtype=int_dtype)
        with pytest.raises(NotImplementedError):
            torchscience.special_functions.gamma(x_int)

    def test_gradient_at_positive_integers(self):
        """Test gradients at positive integers are finite."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.gamma(x)
        y.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_gradient_near_poles_finite(self):
        """Test that gradients near (but not at) poles are finite."""
        x = torch.tensor(
            [0.001, -0.999, -1.999], dtype=torch.float64, requires_grad=True
        )
        y = torchscience.special_functions.gamma(x)
        y.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_complex_gradient_at_poles_nan(self):
        """Test that gradients at complex poles return NaN."""
        for n in [0, -1, -2, -3]:
            x = torch.tensor(
                [complex(n, 0)], dtype=torch.complex128, requires_grad=True
            )
            y = torchscience.special_functions.gamma(x)
            y.real.backward()
            assert torch.isnan(x.grad).all(), (
                f"Expected NaN gradient at {n}+0j"
            )

    def test_complex_gradient_with_imaginary_part_finite(self):
        """Test that gradients at complex values with nonzero imaginary part are finite."""
        x = torch.tensor(
            [
                1.0 + 1.0j,
                2.0 + 0.5j,
                -0.5 + 1.0j,
                -1.0 + 1.0j,
            ],
            dtype=torch.complex128,
            requires_grad=True,
        )
        y = torchscience.special_functions.gamma(x)
        y.real.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_large_negative_integer_poles(self):
        """Test that large negative integers are correctly detected as poles."""
        large_poles = [-100.0, -500.0]
        for pole in large_poles:
            x = torch.tensor([pole], dtype=torch.float64)
            result = torchscience.special_functions.gamma(x)
            assert torch.isinf(result).all(), (
                f"Expected inf at {pole}, got {result}"
            )
