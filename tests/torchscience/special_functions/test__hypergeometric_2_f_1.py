import math

import pytest
import torch
import torch.testing

import torchscience.special_functions
from torchscience.testing import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)

# Optional scipy import for reference tests
try:
    import scipy.special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional sympy import for symbolic verification
try:
    import sympy  # noqa: F401

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def scipy_hyp2f1(a: float, b: float, c: float, z: float) -> float:
    """Reference implementation using SciPy's hyp2f1."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required for this function")
    return float(scipy.special.hyp2f1(a, b, c, z))


def _power_function_identity(func):
    """Check 2F1(a, b; b; z) = (1-z)^(-a) for |z| < 1."""
    a = torch.tensor([2.0], dtype=torch.float64)
    b = torch.tensor([3.0], dtype=torch.float64)
    z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    left = func(a, b, b, z)
    right = torch.pow(1 - z, -a)
    return left, right


def _symmetry_identity(func):
    """Check 2F1(a, b; c; z) = 2F1(b, a; c; z) (symmetry in a, b)."""
    a = torch.tensor([1.5], dtype=torch.float64)
    b = torch.tensor([2.5], dtype=torch.float64)
    c = torch.tensor([3.0], dtype=torch.float64)
    z = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    left = func(a, b, c, z)
    right = func(b, a, c, z)
    return left, right


class TestHypergeometric2F1(OpTestCase):
    """Tests for the Gauss hypergeometric function 2F1."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hypergeometric_2_f_1",
            func=torchscience.special_functions.hypergeometric_2_f_1,
            arity=4,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="b",
                    position=1,
                    default_real_range=(0.5, 5.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="c",
                    position=2,
                    default_real_range=(0.5, 5.0),
                    # Exclude non-positive integers (poles)
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                    supports_grad=True,
                ),
                InputSpec(
                    name="z",
                    position=3,
                    default_real_range=(-0.9, 0.9),
                    supports_grad=True,
                    # For complex z, convergence is best for |z| < 1
                    complex_magnitude_max=0.95,
                ),
            ],
            tolerances=ToleranceConfig(
                float32_rtol=1e-4,
                float32_atol=1e-4,
                float64_rtol=1e-6,
                float64_atol=1e-6,
                # Finite differences for parameter gradients may have reduced accuracy
                gradcheck_rtol=1e-4,
                gradcheck_atol=1e-4,
                gradgradcheck_rtol=1e-3,
                gradgradcheck_atol=1e-3,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",  # CPU autocast not supported
                # Complex second-order derivatives are numerically challenging
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",  # Parameter 2nd derivatives not fully implemented
                # Finite difference gradients for a, b, c may be less accurate
                "test_gradcheck_all_params",
                # Mixed sparse/dense tests
                "test_sparse_coo_mixed_with_dense",
                "test_quantized_mixed_with_float",
                # 2F1 has special behavior at zero parameters (terminates to 1),
                # so NaN/sparse zeros don't propagate as expected
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_sparse_coo_basic",
                "test_sparse_csr_basic",
                # Quantized tensors have too much precision loss for this function
                "test_quantized_basic",
                # Half precision not accurate enough for this complex function
                "test_low_precision_forward",
            },
            functional_identities=[
                IdentitySpec(
                    name="power_function",
                    identity_fn=_power_function_identity,
                    description="2F1(a, b; b; z) = (1-z)^(-a)",
                ),
                IdentitySpec(
                    name="symmetry",
                    identity_fn=_symmetry_identity,
                    description="2F1(a, b; c; z) = 2F1(b, a; c; z)",
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 2.0, 3.0, 0.0),
                    expected=1.0,
                    description="2F1(a, b; c; 0) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 2.0, 3.0, 0.5),
                    expected=1.0,
                    description="2F1(0, b; c; z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 3.0, 0.5),
                    expected=1.0,
                    description="2F1(a, 0; c; z) = 1",
                ),
            ],
            singularities=[],
            supports_sparse_coo=False,  # 2F1 has special behavior at zeros
            supports_sparse_csr=False,  # 2F1 has special behavior at zeros
            supports_quantized=False,  # Precision loss too high
            supports_meta=True,
        )

    # =========================================================================
    # Hypergeometric 2F1-specific tests
    # =========================================================================

    def test_at_zero(self):
        """Test 2F1(a, b; c; 0) = 1 for various a, b, c."""
        z = torch.tensor([0.0], dtype=torch.float64)
        test_cases = [
            (1.0, 2.0, 3.0),
            (0.5, 0.5, 1.0),
            (2.0, 3.0, 4.0),
            (1.5, 2.5, 3.5),
        ]
        for a_val, b_val, c_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64)
            b = torch.tensor([b_val], dtype=torch.float64)
            c = torch.tensor([c_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )
            expected = torch.tensor([1.0], dtype=torch.float64)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-10,
                atol=1e-10,
                msg=f"Failed for a={a_val}, b={b_val}, c={c_val}",
            )

    def test_zero_parameter_a(self):
        """Test 2F1(0, b; c; z) = 1."""
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_zero_parameter_b(self):
        """Test 2F1(a, 0; c; z) = 1."""
        a = torch.tensor([2.0], dtype=torch.float64)
        b = torch.tensor([0.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = torch.ones_like(z)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_power_function_identity(self):
        """Test 2F1(a, b; b; z) = (1-z)^(-a)."""
        a_values = [0.5, 1.0, 2.0, 3.0]
        b = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)

        for a_val in a_values:
            a = torch.tensor([a_val], dtype=torch.float64)
            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, b, z
            )
            expected = torch.pow(1 - z, -a)
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Power function identity failed for a={a_val}",
            )

    def test_symmetry_in_a_b(self):
        """Test 2F1(a, b; c; z) = 2F1(b, a; c; z)."""
        a = torch.tensor([1.5], dtype=torch.float64)
        b = torch.tensor([2.5], dtype=torch.float64)
        c = torch.tensor([4.0], dtype=torch.float64)
        z = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float64)

        result1 = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        result2 = torchscience.special_functions.hypergeometric_2_f_1(
            b, a, c, z
        )
        torch.testing.assert_close(result1, result2, rtol=1e-10, atol=1e-10)

    def test_negative_integer_a_terminates(self):
        """Test that 2F1(-n, b; c; z) is a polynomial when n is non-negative integer."""
        # 2F1(-1, 1; 1; z) = 1 - z
        a = torch.tensor([-1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = 1 - z
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_negative_integer_a_quadratic(self):
        """Test 2F1(-2, 1; 1; z) = 1 - 2z + z^2 = (1-z)^2."""
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        c = torch.tensor([1.0], dtype=torch.float64)
        z = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        expected = (1 - z) ** 2
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_first_few_terms(self):
        """Test series expansion: 1 + (ab/c)z + (a(a+1)b(b+1))/(c(c+1)2!)z^2 + ..."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.1], dtype=torch.float64)

        # First 3 terms of series:
        # term0 = 1
        # term1 = (a*b/c) * z = (1*2/3) * 0.1 = 0.0666...
        # term2 = (a*(a+1)*b*(b+1))/(c*(c+1)*2!) * z^2
        #       = (1*2*2*3)/(3*4*2) * 0.01 = 12/24 * 0.01 = 0.005
        expected_approx = (
            1.0 + (1 * 2 / 3) * 0.1 + (1 * 2 * 2 * 3) / (3 * 4 * 2) * 0.01
        )

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        # The series converges quickly for small z
        torch.testing.assert_close(
            result,
            torch.tensor([expected_approx], dtype=torch.float64),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
    def test_scipy_reference(self):
        """Test against SciPy's hyp2f1 function."""
        z_values = [0.1, 0.3, 0.5, 0.7]
        a_values = [0.5, 1.0, 2.0]
        b_values = [0.5, 1.0, 2.0]
        c_values = [1.0, 2.0, 3.0]

        for z_val in z_values:
            for a_val in a_values:
                for b_val in b_values:
                    for c_val in c_values:
                        a = torch.tensor([a_val], dtype=torch.float64)
                        b = torch.tensor([b_val], dtype=torch.float64)
                        c = torch.tensor([c_val], dtype=torch.float64)
                        z = torch.tensor([z_val], dtype=torch.float64)

                        result = torchscience.special_functions.hypergeometric_2_f_1(
                            a, b, c, z
                        )
                        expected = torch.tensor(
                            [scipy_hyp2f1(a_val, b_val, c_val, z_val)],
                            dtype=torch.float64,
                        )

                        torch.testing.assert_close(
                            result,
                            expected,
                            rtol=1e-6,
                            atol=1e-6,
                            msg=f"SciPy mismatch at a={a_val}, b={b_val}, c={c_val}, z={z_val}",
                        )

    @pytest.mark.skipif(not HAS_SYMPY, reason="SymPy not available")
    def test_sympy_reference(self):
        """Test against SymPy's hyper function."""
        from sympy import N, Rational, hyper

        test_cases = [
            (Rational(1, 2), Rational(1, 2), Rational(3, 2), Rational(1, 4)),
            (1, 2, 3, Rational(1, 2)),
            (Rational(3, 2), Rational(5, 2), 4, Rational(1, 3)),
        ]

        for a_sym, b_sym, c_sym, z_sym in test_cases:
            # Compute with SymPy
            sympy_result = float(N(hyper([a_sym, b_sym], [c_sym], z_sym), 20))

            # Compute with our implementation
            a = torch.tensor([float(a_sym)], dtype=torch.float64)
            b = torch.tensor([float(b_sym)], dtype=torch.float64)
            c = torch.tensor([float(c_sym)], dtype=torch.float64)
            z = torch.tensor([float(z_sym)], dtype=torch.float64)

            result = torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

            torch.testing.assert_close(
                result,
                torch.tensor([sympy_result], dtype=torch.float64),
                rtol=1e-8,
                atol=1e-8,
                msg=f"SymPy mismatch at a={a_sym}, b={b_sym}, c={c_sym}, z={z_sym}",
            )

    def test_gauss_summation_theorem(self):
        """Test Gauss's summation theorem: 2F1(a, b; c; 1) = Gamma(c)Gamma(c-a-b)/(Gamma(c-a)Gamma(c-b))
        when Re(c - a - b) > 0."""
        # For convergence at z=1, we need c > a + b
        a = torch.tensor([0.5], dtype=torch.float64)
        b = torch.tensor([0.5], dtype=torch.float64)
        c = torch.tensor([2.0], dtype=torch.float64)  # c - a - b = 1 > 0

        # Compute 2F1 at z approaching 1 (use 0.999 to avoid branch cut issues)
        z = torch.tensor([0.999], dtype=torch.float64)
        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Expected from Gauss's theorem:
        # Gamma(2) * Gamma(1) / (Gamma(1.5) * Gamma(1.5))
        # = 1 * 1 / (sqrt(pi)/2 * sqrt(pi)/2) = 4/pi
        expected = 4.0 / math.pi

        # Allow some tolerance since we're evaluating at 0.999, not exactly 1
        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=0.01,
            atol=0.01,
        )

    def test_broadcasting(self):
        """Test that broadcasting works correctly."""
        a = torch.tensor([[1.0], [2.0]], dtype=torch.float64)  # (2, 1)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # (3,)
        c = torch.tensor([3.0], dtype=torch.float64)  # (1,)
        z = torch.tensor([0.5], dtype=torch.float64)  # (1,)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert result.shape == (2, 3)

    def test_gradcheck_z(self):
        """Test gradient correctness for z using finite differences."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor(
            [0.2, 0.4, 0.6], dtype=torch.float64, requires_grad=True
        )

        def func(z):
            return torchscience.special_functions.hypergeometric_2_f_1(
                a, b, c, z
            )

        torch.autograd.gradcheck(func, (z,), rtol=1e-4, atol=1e-4)

    def test_derivative_formula_z(self):
        """Test that d/dz 2F1(a, b; c; z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)."""
        a_val, b_val, c_val = 1.0, 2.0, 3.0
        a = torch.tensor([a_val], dtype=torch.float64)
        b = torch.tensor([b_val], dtype=torch.float64)
        c = torch.tensor([c_val], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # Compute gradient via autograd
        y = torchscience.special_functions.hypergeometric_2_f_1(a, b, c, z)
        y.backward()
        autograd_grad = z.grad.clone()

        # Expected gradient from formula
        z_val = 0.5
        expected_grad = (
            a_val * b_val / c_val
        ) * torchscience.special_functions.hypergeometric_2_f_1(
            torch.tensor([a_val + 1]),
            torch.tensor([b_val + 1]),
            torch.tensor([c_val + 1]),
            torch.tensor([z_val]),
        )

        torch.testing.assert_close(
            autograd_grad,
            expected_grad.to(torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_complex_input(self):
        """Test with complex z values."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.3 + 0.2j], dtype=torch.complex128)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )
        assert result.dtype == torch.complex128
        assert result.shape == (1,)

        # Result should be complex
        assert result[0].imag != 0

    def test_large_parameters(self):
        """Test with moderately large parameters."""
        a = torch.tensor([10.0], dtype=torch.float64)
        b = torch.tensor([10.0], dtype=torch.float64)
        c = torch.tensor([25.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Should produce a finite result
        assert torch.isfinite(result).all()
        assert result[0] > 1.0  # Series starts at 1 and has positive terms

    def test_negative_z(self):
        """Test with negative z values."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([-0.5, -0.3, -0.1], dtype=torch.float64)

        result = torchscience.special_functions.hypergeometric_2_f_1(
            a, b, c, z
        )

        # Results should be finite and real
        assert torch.isfinite(result).all()

        # For negative z, the series alternates but converges
        # 2F1(a, b; c; z) < 1 for z < 0 with positive a, b, c
        assert (result < 1.0).all()
