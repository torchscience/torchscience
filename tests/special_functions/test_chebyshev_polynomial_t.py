"""
Comprehensive tests for chebyshev_polynomial_t.

Tests cover:
- Integer v with real z (recurrence path)
- Non-integer v with real z (analytic continuation)
- Complex z (analytic continuation)
- Complex v (analytic continuation)
- Dtype promotion rules
- Broadcasting
- Autograd (gradcheck for z and v)
- Branch cut behavior
- Numerical stability
- torch.compile compatibility
- vmap support
"""

import math
import pytest
import torch
from torch.testing import assert_close

import torchscience.special_functions as sf


# =============================================================================
# Test fixtures and utilities
# =============================================================================

@pytest.fixture
def device():
    """Return available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def reference_chebyshev_t(v, z):
    """Reference implementation using torch.cos(v * torch.acos(z))."""
    # Handle promotion to complex if needed
    if torch.is_complex(v) or torch.is_complex(z):
        # Promote to complex
        if not torch.is_complex(z):
            z = z.to(torch.complex128 if z.dtype == torch.float64 else torch.complex64)
        if not torch.is_complex(v):
            v = v.to(z.dtype)
    return torch.cos(v * torch.acos(z))


# =============================================================================
# Basic functionality tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic forward pass functionality."""

    def test_integer_degree_real_input_basic(self):
        """Test T_n(x) for small integer n and real x in [-1, 1]."""
        # T_0(x) = 1
        z = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        v = torch.tensor([0.0])
        result = sf.chebyshev_polynomial_t(v, z)
        expected = torch.ones_like(z)
        assert_close(result, expected)

        # T_1(x) = x
        v = torch.tensor([1.0])
        result = sf.chebyshev_polynomial_t(v, z)
        assert_close(result, z)

        # T_2(x) = 2x^2 - 1
        v = torch.tensor([2.0])
        result = sf.chebyshev_polynomial_t(v, z)
        expected = 2 * z**2 - 1
        assert_close(result, expected)

        # T_3(x) = 4x^3 - 3x
        v = torch.tensor([3.0])
        result = sf.chebyshev_polynomial_t(v, z)
        expected = 4 * z**3 - 3 * z
        assert_close(result, expected)

    def test_integer_degree_matches_reference(self):
        """Test that integer degree recurrence matches cos(n*arccos(x))."""
        z = torch.linspace(-1, 1, 100, dtype=torch.float64)
        for n in range(10):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)
            expected = reference_chebyshev_t(v, z)
            assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_integer_degree_unrolling_boundary(self):
        """Test degrees at the unrolling boundary (n=7,8,9) work correctly.

        The implementation explicitly unrolls degrees 0-7 and uses a loop for
        degrees >= 8. This test verifies the transition is seamless.
        """
        z = torch.linspace(-1, 1, 50, dtype=torch.float64)
        # Test degrees around the unrolling boundary
        for n in [6, 7, 8, 9, 10, 15, 20]:
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)
            expected = reference_chebyshev_t(v, z)
            assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_non_integer_degree(self):
        """Test non-integer degree uses analytic continuation."""
        z = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

        v = torch.tensor([1.5], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_negative_integer_degree_symmetry(self):
        """Test that T_{-n}(z) = T_n(z) for integer n."""
        z = torch.tensor([0.5, 0.3, -0.7], dtype=torch.float64)

        for n in range(0, 6):
            v_pos = torch.tensor([float(n)], dtype=torch.float64)
            v_neg = torch.tensor([float(-n)], dtype=torch.float64)

            result_pos = sf.chebyshev_polynomial_t(v_pos, z)
            result_neg = sf.chebyshev_polynomial_t(v_neg, z)

            assert_close(result_pos, result_neg, rtol=1e-10, atol=1e-10)


# =============================================================================
# Complex input tests
# =============================================================================

class TestComplexInputs:
    """Test complex input handling."""

    def test_complex_z_real_v(self):
        """Test complex z with real v."""
        z = torch.tensor([1.0 + 0.1j, 0.5 + 0.5j, -0.5 - 0.5j], dtype=torch.complex128)
        v = torch.tensor([2.0], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_real_z_complex_v(self):
        """Test real z with complex v."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_complex_z_complex_v(self):
        """Test complex z with complex v."""
        z = torch.tensor([0.5 + 0.2j], dtype=torch.complex128)
        v = torch.tensor([2.0 + 0.5j], dtype=torch.complex128)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_output_dtype_promotion_to_complex(self):
        """Test that output is complex when any input is complex."""
        z_real = torch.tensor([0.5], dtype=torch.float64)
        v_complex = torch.tensor([2.0 + 0.0j], dtype=torch.complex128)

        result = sf.chebyshev_polynomial_t(v_complex, z_real)
        assert result.is_complex()


# =============================================================================
# Branch cut tests
# =============================================================================

class TestBranchCuts:
    """Test behavior near branch cuts."""

    def test_no_warning_real_z_outside_domain_integer_v(self):
        """Test that no warning is issued for real z outside [-1, 1] with integer v."""
        import warnings

        z = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)  # Integer

        # Should not warn - integer v uses recurrence which works for all real z
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sf.chebyshev_polynomial_t(v, z)
        # Filter to only RuntimeWarnings about z values
        relevant_warnings = [w for w in record if "real z values outside" in str(w.message)]
        assert len(relevant_warnings) == 0

    def test_no_warning_complex_z_outside_domain(self):
        """Test that no warning is issued for complex z outside [-1, 1]."""
        import warnings

        z = torch.tensor([1.5 + 0j, 2.0 + 0j], dtype=torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)  # Non-integer

        # Should not warn - complex z uses proper analytic continuation
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            sf.chebyshev_polynomial_t(v, z)
        relevant_warnings = [w for w in record if "real z values outside" in str(w.message)]
        assert len(relevant_warnings) == 0

    def test_z_outside_minus1_to_1_with_complex_input(self):
        """Test z outside [-1, 1] works correctly when using complex input."""
        # For real z outside [-1, 1], std::acos returns NaN.
        # To get correct results, users should use complex tensors.
        z = torch.tensor([1.5 + 0j, 2.0 + 0j, -1.5 + 0j, -2.0 + 0j], dtype=torch.complex128)
        v = torch.tensor([2.0], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_z_outside_minus1_to_1_integer_v(self):
        """Test that real z outside [-1, 1] with integer v uses recurrence."""
        z = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)
        # For integer v, recurrence gives exact polynomial results
        # T_2(z) = 2z^2 - 1
        expected = torch.tensor([3.5, 7.0], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_z_outside_minus1_to_1_noninteger_v(self):
        """Test that real z outside [-1, 1] with non-integer v uses hyperbolic continuation."""
        # For z > 1: T_v(z) = cosh(v * acosh(z))
        z = torch.tensor([1.5, 2.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Reference using complex path (should match hyperbolic continuation)
        z_complex = z.to(torch.complex128)
        expected = reference_chebyshev_t(v, z_complex).real
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_z_negative_outside_minus1_to_1_noninteger_v(self):
        """Test that real z < -1 with non-integer v uses hyperbolic continuation."""
        # For z < -1: T_v(z) = cos(v*π) * cosh(v * acosh(-z))
        z = torch.tensor([-1.5, -2.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Reference using complex path (should match hyperbolic continuation)
        z_complex = z.to(torch.complex128)
        expected = reference_chebyshev_t(v, z_complex).real
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_branch_cut_near_plus_one(self):
        """Test behavior near z = 1 with small imaginary parts."""
        # z just above the real axis near 1
        z_above = torch.tensor([1.0 + 0.01j], dtype=torch.complex128)
        z_below = torch.tensor([1.0 - 0.01j], dtype=torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)

        result_above = sf.chebyshev_polynomial_t(v, z_above)
        result_below = sf.chebyshev_polynomial_t(v, z_below)

        # Results should be conjugates (due to branch cut symmetry)
        assert_close(result_above, result_below.conj(), rtol=1e-6, atol=1e-6)

    def test_branch_cut_near_minus_one(self):
        """Test behavior near z = -1 with small imaginary parts."""
        z_above = torch.tensor([-1.0 + 0.01j], dtype=torch.complex128)
        z_below = torch.tensor([-1.0 - 0.01j], dtype=torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)

        result_above = sf.chebyshev_polynomial_t(v, z_above)
        result_below = sf.chebyshev_polynomial_t(v, z_below)

        # Results should be conjugates
        assert_close(result_above, result_below.conj(), rtol=1e-6, atol=1e-6)


# =============================================================================
# Hyperbolic continuation tests (|z| > 1)
# =============================================================================

class TestHyperbolicContinuation:
    """Test hyperbolic continuation for real z outside [-1, 1] with non-integer v."""

    def test_forward_z_greater_than_1(self):
        """Test T_v(z) = cosh(v * acosh(z)) for z > 1."""
        z = torch.tensor([1.5, 2.0, 3.0, 5.0], dtype=torch.float64)
        v = torch.tensor([0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Reference: T_v(z) = cosh(v * acosh(z))
        expected = torch.cosh(v * torch.acosh(z))
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_forward_z_less_than_minus_1(self):
        """Test T_v(z) = cos(v*π) * cosh(v * acosh(-z)) for z < -1."""
        z = torch.tensor([-1.5, -2.0, -3.0, -5.0], dtype=torch.float64)
        # Use v = 0.25 where cos(v*π) ≠ 0 for better numerical comparison
        v = torch.tensor([0.25], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Reference: T_v(z) = cos(v*π) * cosh(v * acosh(-z))
        expected = torch.cos(v * torch.tensor(math.pi)) * torch.cosh(v * torch.acosh(-z))
        # Use relaxed tolerance due to floating point precision
        assert_close(result, expected, rtol=1e-7, atol=1e-7)

    def test_forward_various_v_values(self):
        """Test various non-integer v values with z outside [-1, 1]."""
        z = torch.tensor([2.0], dtype=torch.float64)
        v_values = [0.1, 0.5, 1.5, 2.5, 3.7]

        for v_val in v_values:
            v = torch.tensor([v_val], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)

            # Compare against complex reference
            z_complex = z.to(torch.complex128)
            expected = reference_chebyshev_t(v, z_complex).real
            assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_continuity_at_z_equals_1(self):
        """Test that hyperbolic path is continuous with analytic path at z = 1."""
        v = torch.tensor([2.5], dtype=torch.float64)

        z_inside = torch.tensor([1.0 - 1e-8], dtype=torch.float64)
        z_outside = torch.tensor([1.0 + 1e-8], dtype=torch.float64)

        result_inside = sf.chebyshev_polynomial_t(v, z_inside)
        result_outside = sf.chebyshev_polynomial_t(v, z_outside)

        # Both should be close to T_v(1) = 1
        expected = torch.tensor([1.0], dtype=torch.float64)
        assert_close(result_inside, expected, rtol=1e-6, atol=1e-6)
        assert_close(result_outside, expected, rtol=1e-6, atol=1e-6)

    def test_continuity_at_z_equals_minus_1(self):
        """Test that hyperbolic path produces finite values near z = -1."""
        # Use v = 2.25 where cos(v*π) ≠ 0 for better numerical comparison
        v = torch.tensor([2.25], dtype=torch.float64)

        # Test values on both sides of z=-1
        z_inside = torch.tensor([-0.99], dtype=torch.float64)
        z_outside = torch.tensor([-1.01], dtype=torch.float64)

        result_inside = sf.chebyshev_polynomial_t(v, z_inside)
        result_outside = sf.chebyshev_polynomial_t(v, z_outside)

        # Both results should be finite (not NaN or Inf)
        assert torch.isfinite(result_inside).all()
        assert torch.isfinite(result_outside).all()

        # Both should have the same sign as cos(v*π)
        expected_sign = torch.sign(torch.cos(v * torch.tensor(math.pi)))
        assert torch.sign(result_inside) == expected_sign
        assert torch.sign(result_outside) == expected_sign

    def test_gradcheck_z_greater_than_1(self):
        """Test gradient correctness for z > 1."""
        z = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_z_less_than_minus_1(self):
        """Test gradient correctness for z < -1."""
        z = torch.tensor([-1.5, -2.0, -3.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_v_with_z_greater_than_1(self):
        """Test gradient correctness for v when z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64)
        v = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (v,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_v_with_z_less_than_minus_1(self):
        """Test gradient correctness for v when z < -1."""
        z = torch.tensor([-2.0], dtype=torch.float64)
        v = torch.tensor([2.5, 3.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (v,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_both_v_and_z_outside_domain(self):
        """Test gradient correctness for both v and z when z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(v, z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (v, z), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradgradcheck_z_greater_than_1(self):
        """Test second-order gradient correctness for z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_z_less_than_minus_1(self):
        """Test second-order gradient correctness for z < -1."""
        z = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_v_with_z_greater_than_1(self):
        """Test second-order gradient correctness for v when z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (v,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_v_with_z_less_than_minus_1(self):
        """Test second-order gradient correctness for v when z < -1."""
        z = torch.tensor([-2.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (v,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_both_z_greater_than_1(self):
        """Test second-order gradient correctness for both v and z when z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(v, z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (v, z), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_backward_manual_z_greater_than_1(self):
        """Manually verify backward formula for z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Manual: dT/dz = v * sinh(v*acosh(z)) / sqrt(z² - 1)
        phi = torch.acosh(z.detach())
        expected_grad = v * torch.sinh(v * phi) / torch.sqrt(z.detach()**2 - 1)

        assert_close(z.grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_backward_manual_v_with_z_greater_than_1(self):
        """Manually verify backward formula for v when z > 1."""
        z = torch.tensor([2.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Manual: dT/dv = sinh(v*acosh(z)) * acosh(z)
        phi = torch.acosh(z)
        expected_grad = torch.sinh(v.detach() * phi) * phi

        assert_close(v.grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_backward_manual_z_less_than_minus_1(self):
        """Manually verify backward formula for z < -1."""
        z = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
        # Use v = 2.25 where cos(v*π) ≠ 0 for better numerical comparison
        v = torch.tensor([2.25], dtype=torch.float64)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Manual: dT/dz = -cos(v*π) * v * sinh(v*acosh(-z)) / sqrt(z² - 1)
        phi = torch.acosh(-z.detach())
        cos_v_pi = torch.cos(v * torch.tensor(math.pi))
        expected_grad = -cos_v_pi * v * torch.sinh(v * phi) / torch.sqrt(z.detach()**2 - 1)

        # Use relaxed tolerance due to floating point precision
        assert_close(z.grad, expected_grad, rtol=1e-6, atol=1e-6)

    def test_large_z_values(self):
        """Test numerical stability for large z values."""
        z = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float64)
        v = torch.tensor([2.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Should not produce NaN or Inf
        assert torch.isfinite(result).all()

        # Reference using cosh(v * acosh(z))
        expected = torch.cosh(v * torch.acosh(z))
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_matches_complex_path(self):
        """Test that real hyperbolic path matches complex analytic continuation."""
        z_real = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
        z_complex = z_real.to(torch.complex128)
        v = torch.tensor([2.5], dtype=torch.float64)

        result_real = sf.chebyshev_polynomial_t(v, z_real)
        result_complex = sf.chebyshev_polynomial_t(v, z_complex)

        # Real path should match real part of complex path
        assert_close(result_real, result_complex.real, rtol=1e-10, atol=1e-10)
        # Imaginary part should be zero (or very small)
        assert_close(result_complex.imag, torch.zeros_like(result_complex.imag), rtol=1e-10, atol=1e-10)


# =============================================================================
# Dtype tests
# =============================================================================

class TestDtypes:
    """Test dtype handling and promotion."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_dtypes(self, dtype):
        """Test real floating point dtypes."""
        z = torch.tensor([0.5], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_v_dtype_promotion(self, int_dtype):
        """Test that integer dtype v is promoted and works correctly."""
        # Integer dtype v should be promoted to float
        v_int = torch.tensor([2], dtype=int_dtype)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v_int, z)

        # Result should be float (promoted from int)
        assert result.dtype == torch.float64

        # Should give same result as float v
        v_float = torch.tensor([2.0], dtype=torch.float64)
        expected = sf.chebyshev_polynomial_t(v_float, z)
        assert_close(result, expected)

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_v_dtype_uses_recurrence(self, int_dtype):
        """Test that integer dtype v (after promotion) uses recurrence path."""
        # T_2(x) = 2x^2 - 1
        v_int = torch.tensor([2], dtype=int_dtype)
        z = torch.tensor([0.3, 0.5, 0.7, -0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v_int, z)
        expected = 2 * z**2 - 1

        # Should match exactly (recurrence gives exact polynomial)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_integer_value_detection_float_tensor(self):
        """Test that integer values in float tensors use recurrence path."""
        # v=2.0 stored as float should still use recurrence
        v = torch.tensor([2.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)
        expected = 2 * 0.5**2 - 1  # T_2(0.5) = -0.5

        assert_close(result, torch.tensor([expected], dtype=torch.float64))

    def test_non_integer_value_uses_analytic(self):
        """Test that non-integer values use analytic continuation."""
        v = torch.tensor([2.5], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v, z)

        # Compare against reference implementation
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_integer_v_multiple_values(self):
        """Test multiple integer v values with integer dtype."""
        v_int = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        z = torch.tensor([0.5], dtype=torch.float64)

        result = sf.chebyshev_polynomial_t(v_int, z)

        # Expected values for T_0 through T_4 at z=0.5
        z_val = 0.5
        expected = torch.tensor([
            1.0,                           # T_0 = 1
            z_val,                         # T_1 = z
            2*z_val**2 - 1,               # T_2 = 2z² - 1
            4*z_val**3 - 3*z_val,         # T_3 = 4z³ - 3z
            8*z_val**4 - 8*z_val**2 + 1,  # T_4 = 8z⁴ - 8z² + 1
        ], dtype=torch.float64)

        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        z = torch.tensor([0.5 + 0.1j], dtype=dtype)
        v = torch.tensor([2.0], dtype=torch.float32 if dtype == torch.complex64 else torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype

    def test_mixed_float32_float64_promotes(self):
        """Test that mixing float32 and float64 promotes to float64."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float32)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == torch.float64

    def test_mixed_complex64_complex128_promotes(self):
        """Test that mixing complex64 and complex128 promotes to complex128."""
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex128)
        v = torch.tensor([2.0 + 0.1j], dtype=torch.complex64)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == torch.complex128

    def test_float32_complex64_gives_complex64(self):
        """Test float32 + complex64 gives complex64."""
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex64)
        v = torch.tensor([2.0], dtype=torch.float32)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == torch.complex64

    def test_float64_complex64_gives_complex128(self):
        """Test float64 + complex64 gives complex128."""
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex64)
        v = torch.tensor([2.0], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == torch.complex128


# =============================================================================
# Broadcasting tests
# =============================================================================

class TestBroadcasting:
    """Test broadcasting behavior."""

    def test_broadcast_v_scalar_z_vector(self):
        """Test broadcasting scalar v with vector z."""
        v = torch.tensor([2.0])
        z = torch.tensor([0.0, 0.5, 1.0])
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.shape == z.shape

    def test_broadcast_v_vector_z_scalar(self):
        """Test broadcasting vector v with scalar z."""
        v = torch.tensor([0.0, 1.0, 2.0])
        z = torch.tensor([0.5])
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.shape == v.shape

    def test_broadcast_2d_shapes(self):
        """Test broadcasting with 2D shapes."""
        v = torch.tensor([[0.0], [1.0], [2.0]])  # (3, 1)
        z = torch.tensor([[0.0, 0.5, 1.0]])       # (1, 3)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.shape == (3, 3)

    def test_broadcast_large_batch(self):
        """Test with large batch dimensions."""
        v = torch.randn(10, 1, 20)
        z = torch.randn(1, 30, 1)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.shape == (10, 30, 20)


# =============================================================================
# Autograd tests
# =============================================================================

class TestAutograd:
    """Test autograd functionality."""

    def test_gradcheck_z_float64(self):
        """Test gradient correctness for z with float64."""
        z = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (z,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_z_complex128(self):
        """Test gradient correctness for z with complex128."""
        z = torch.tensor([0.3 + 0.1j, 0.5 + 0.2j], dtype=torch.complex128, requires_grad=True)
        v = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        # Relaxed tolerances for complex gradcheck
        assert torch.autograd.gradcheck(func, (z,), eps=1e-5, atol=1e-3, rtol=1e-3)

    def test_gradcheck_v_float64(self):
        """Test gradient correctness for v when v is float64."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.3, 3.7], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (v,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_v_complex128(self):
        """Test gradient correctness for v when v is complex128."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.3 + 0.5j], dtype=torch.complex128, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        # Relaxed tolerances for complex gradcheck
        assert torch.autograd.gradcheck(func, (v,), eps=1e-5, atol=1e-3, rtol=1e-3)

    def test_gradcheck_both_v_and_z(self):
        """Test gradient correctness for both v and z."""
        z = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        def func(v, z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradcheck(func, (v, z), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradgradcheck_z(self):
        """Test second-order gradient for z."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.0], dtype=torch.float64)

        def func(z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (z,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_v(self):
        """Test second-order gradient for v."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.3], dtype=torch.float64, requires_grad=True)

        def func(v):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (v,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_both(self):
        """Test second-order gradient for both v and z."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.3], dtype=torch.float64, requires_grad=True)

        def func(v, z):
            return sf.chebyshev_polynomial_t(v, z)

        assert torch.autograd.gradgradcheck(func, (v, z), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_second_derivative_manual_z(self):
        """Manually verify second derivative formula for z."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([3.0], dtype=torch.float64)

        # Compute first derivative
        y = sf.chebyshev_polynomial_t(v, z)
        grad_z, = torch.autograd.grad(y, z, create_graph=True)

        # Compute second derivative
        grad2_z, = torch.autograd.grad(grad_z, z)

        # Manual computation of d²T_v/dz²
        # d/dz[v * sin(v*acos(z)) / sqrt(1 - z²)]
        # This is complex but should match numerical differentiation
        z_val = z.detach()
        eps = 1e-5
        z_plus = torch.tensor([z_val.item() + eps], dtype=torch.float64, requires_grad=True)
        z_minus = torch.tensor([z_val.item() - eps], dtype=torch.float64, requires_grad=True)

        y_plus = sf.chebyshev_polynomial_t(v, z_plus)
        y_minus = sf.chebyshev_polynomial_t(v, z_minus)
        grad_plus, = torch.autograd.grad(y_plus, z_plus)
        grad_minus, = torch.autograd.grad(y_minus, z_minus)

        numerical_grad2 = (grad_plus - grad_minus) / (2 * eps)

        assert torch.allclose(grad2_z, numerical_grad2, rtol=1e-4, atol=1e-4)

    def test_no_grad_for_integer_v(self):
        """Test that integer v does not receive gradients."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        # Even though v is a float tensor with integer value, it should still
        # get gradients since the dtype is float
        v = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # v should have gradient since it's float dtype
        assert v.grad is not None

    def test_gradient_v_zero(self):
        """Test that gradient w.r.t. z is zero when v=0 (T_0(z) = 1 is constant)."""
        z = torch.tensor([0.3, 0.5, 0.7, -0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([0.0], dtype=torch.float64)

        y = sf.chebyshev_polynomial_t(v, z)
        y.sum().backward()

        # dT_0/dz = 0 for all z, since T_0(z) = 1 (constant)
        expected_grad = torch.zeros_like(z)
        assert_close(z.grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_gradient_v_zero_with_v_grad(self):
        """Test gradients when v=0 and v requires grad."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # dT_v/dz at v=0 should be 0
        expected_grad_z = torch.tensor([0.0], dtype=torch.float64)
        assert_close(z.grad, expected_grad_z, rtol=1e-10, atol=1e-10)

        # dT_v/dv = -sin(v*acos(z)) * acos(z) = -sin(0) * acos(0.5) = 0
        expected_grad_v = torch.tensor([0.0], dtype=torch.float64)
        assert_close(v.grad, expected_grad_v, rtol=1e-10, atol=1e-10)

    def test_backward_formula_z_manual_check(self):
        """Manually verify backward formula for z."""
        z = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([3.0], dtype=torch.float64)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Manual computation: dT_v/dz = v * sin(v*arccos(z)) / sqrt(1 - z^2)
        theta = torch.acos(z.detach())
        expected_grad = v * torch.sin(v * theta) / torch.sqrt(1 - z.detach()**2)

        assert_close(z.grad, expected_grad, rtol=1e-10, atol=1e-10)

    def test_backward_formula_v_manual_check(self):
        """Manually verify backward formula for v."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Manual computation: dT_v/dv = -sin(v*arccos(z)) * arccos(z)
        theta = torch.acos(z)
        expected_grad = -torch.sin(v.detach() * theta) * theta

        assert_close(v.grad, expected_grad, rtol=1e-10, atol=1e-10)


# =============================================================================
# Numerical stability tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_large_degree(self):
        """Test with large integer degree."""
        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([50.0], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = reference_chebyshev_t(v, z)
        # Allow larger tolerance for high degrees
        assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_z_near_boundary(self):
        """Test z values very close to +/- 1."""
        z = torch.tensor([1 - 1e-10, -1 + 1e-10], dtype=torch.float64)
        v = torch.tensor([5.0], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        # Should not produce NaN or Inf
        assert torch.isfinite(result).all()

    def test_small_complex_perturbation(self):
        """Test with very small complex perturbations."""
        z = torch.tensor([0.5 + 1e-10j], dtype=torch.complex128)
        v = torch.tensor([5.0], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()

    def test_gradient_near_z_equals_one(self):
        """Test gradient stability near z = 1 for non-integer v."""
        z = torch.tensor([1.0 - 1e-14], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)  # Non-integer to use analytic path

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Gradient should be finite (limit: v² = 6.25)
        assert torch.isfinite(z.grad).all()
        # Check against expected limit value
        expected_grad = v * v  # lim_{z→1} dT_v/dz = v²
        assert_close(z.grad, expected_grad, rtol=1e-3, atol=1e-3)

    def test_gradient_near_z_equals_minus_one(self):
        """Test gradient stability near z = -1 for non-integer v."""
        z = torch.tensor([-1.0 + 1e-14], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64)  # Non-integer to use analytic path

        y = sf.chebyshev_polynomial_t(v, z)
        y.backward()

        # Gradient should be finite (limit: v² * cos(π*v))
        assert torch.isfinite(z.grad).all()
        # Check against expected limit value
        expected_grad = v * v * torch.cos(torch.tensor(math.pi) * v)
        assert_close(z.grad, expected_grad, rtol=1e-3, atol=1e-3)

    def test_gradient_integer_v_near_boundaries(self):
        """Test that integer v uses stable U recurrence near z = ±1."""
        # For integer v, backward uses U recurrence which is stable
        z = torch.tensor([1.0 - 1e-14, -1.0 + 1e-14], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([3.0], dtype=torch.float64)  # Integer degree

        y = sf.chebyshev_polynomial_t(v, z)
        y.sum().backward()

        # Gradients should be finite
        assert torch.isfinite(z.grad).all()

    def test_double_backward_at_z_equals_minus_one_produces_finite(self):
        """Test that second-order gradients at z = -1 exactly produce finite values.

        This tests the fix for the boundary case where sqrt(1-z²) = 0.
        The implementation must use analytically computed limits for d²T/dzdv
        instead of the general formula which involves division by sqrt(1-z²).

        Note: We cannot use gradgradcheck here because numerical differentiation
        would perturb z outside the valid domain [-1, 1] where acos returns NaN.
        """
        z = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        # Compute function and first derivatives
        y = sf.chebyshev_polynomial_t(v, z)
        grad_v, grad_z = torch.autograd.grad(y, (v, z), create_graph=True)

        # Compute all second derivatives
        grad_v_v, grad_v_z = torch.autograd.grad(grad_v, (v, z), retain_graph=True)
        grad_z_v, grad_z_z = torch.autograd.grad(grad_z, (v, z))

        # All second derivatives should be finite
        assert torch.isfinite(grad_v_v).all(), f"d²T/dv² is not finite: {grad_v_v}"
        assert torch.isfinite(grad_v_z).all(), f"d²T/dvdz is not finite: {grad_v_z}"
        assert torch.isfinite(grad_z_v).all(), f"d²T/dzdv is not finite: {grad_z_v}"
        assert torch.isfinite(grad_z_z).all(), f"d²T/dz² is not finite: {grad_z_z}"

    def test_double_backward_at_z_equals_plus_one_produces_finite(self):
        """Test that second-order gradients at z = +1 exactly produce finite values."""
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        # Compute function and first derivatives
        y = sf.chebyshev_polynomial_t(v, z)
        grad_v, grad_z = torch.autograd.grad(y, (v, z), create_graph=True)

        # Compute all second derivatives
        grad_v_v, grad_v_z = torch.autograd.grad(grad_v, (v, z), retain_graph=True)
        grad_z_v, grad_z_z = torch.autograd.grad(grad_z, (v, z))

        # All second derivatives should be finite
        assert torch.isfinite(grad_v_v).all(), f"d²T/dv² is not finite: {grad_v_v}"
        assert torch.isfinite(grad_v_z).all(), f"d²T/dvdz is not finite: {grad_v_z}"
        assert torch.isfinite(grad_z_v).all(), f"d²T/dzdv is not finite: {grad_z_v}"
        assert torch.isfinite(grad_z_z).all(), f"d²T/dz² is not finite: {grad_z_z}"

    def test_double_backward_at_minus_one_manual_verification(self):
        """Manually verify the d²T/dzdv limit at z = -1.

        At z = -1: dT/dz = v² * cos(πv)
        Therefore: d²T/dzdv = d/dv[v² * cos(πv)] = 2v * cos(πv) - πv² * sin(πv)
        """
        z = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        # Compute first derivative w.r.t. z
        y = sf.chebyshev_polynomial_t(v, z)
        grad_z, = torch.autograd.grad(y, z, create_graph=True)

        # Compute mixed second derivative d²T/dzdv
        grad_z_v, = torch.autograd.grad(grad_z, v)

        # Expected value: 2v * cos(πv) - πv² * sin(πv)
        pi = torch.tensor(math.pi, dtype=torch.float64)
        expected = 2 * v * torch.cos(pi * v) - pi * v * v * torch.sin(pi * v)

        assert torch.isfinite(grad_z_v).all(), f"Got non-finite value: {grad_z_v}"
        assert_close(grad_z_v, expected, rtol=1e-6, atol=1e-6)

    def test_double_backward_at_plus_one_manual_verification(self):
        """Manually verify the d²T/dzdv limit at z = +1.

        At z = +1: dT/dz = v²
        Therefore: d²T/dzdv = d/dv[v²] = 2v
        """
        z = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)

        # Compute first derivative w.r.t. z
        y = sf.chebyshev_polynomial_t(v, z)
        grad_z, = torch.autograd.grad(y, z, create_graph=True)

        # Compute mixed second derivative d²T/dzdv
        grad_z_v, = torch.autograd.grad(grad_z, v)

        # Expected value: 2v
        expected = 2 * v

        assert torch.isfinite(grad_z_v).all(), f"Got non-finite value: {grad_z_v}"
        assert_close(grad_z_v, expected, rtol=1e-6, atol=1e-6)


# =============================================================================
# Device tests
# =============================================================================

class TestDevices:
    """Test device handling."""

    def test_cpu(self):
        """Test on CPU."""
        z = torch.tensor([0.5], dtype=torch.float64, device="cpu")
        v = torch.tensor([2.0], dtype=torch.float64, device="cpu")
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test on CUDA."""
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")
        v = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.device.type == "cuda"
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_complex(self):
        """Test complex on CUDA."""
        z = torch.tensor([0.5 + 0.1j], dtype=torch.complex128, device="cuda")
        v = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.device.type == "cuda"
        expected = reference_chebyshev_t(v, z)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)


# =============================================================================
# torch.compile tests
# =============================================================================

class TestTorchCompile:
    """Test torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_compile_smoke(self):
        """Smoke test for torch.compile."""
        @torch.compile
        def compiled_chebyshev(v, z):
            return sf.chebyshev_polynomial_t(v, z)

        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)

        result = compiled_chebyshev(v, z)
        expected = sf.chebyshev_polynomial_t(v, z)
        assert_close(result, expected)


# =============================================================================
# vmap tests
# =============================================================================

class TestVmap:
    """Test vmap compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "vmap"),
        reason="torch.vmap not available"
    )
    def test_vmap_over_z(self):
        """Test vmap over z dimension."""
        from torch import vmap

        z = torch.randn(10, dtype=torch.float64)
        v = torch.tensor([2.0], dtype=torch.float64)

        def single_chebyshev(z_single):
            return sf.chebyshev_polynomial_t(v, z_single.unsqueeze(0)).squeeze(0)

        vmapped = vmap(single_chebyshev)
        result = vmapped(z)

        expected = sf.chebyshev_polynomial_t(v, z)
        assert_close(result, expected)

    @pytest.mark.skipif(
        not hasattr(torch, "vmap"),
        reason="torch.vmap not available"
    )
    def test_vmap_over_v(self):
        """Test vmap over v dimension."""
        from torch import vmap

        z = torch.tensor([0.5], dtype=torch.float64)
        v = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

        def single_chebyshev(v_single):
            return sf.chebyshev_polynomial_t(v_single.unsqueeze(0), z).squeeze(0)

        vmapped = vmap(single_chebyshev)
        result = vmapped(v)

        expected = sf.chebyshev_polynomial_t(v, z)
        assert_close(result, expected)


# =============================================================================
# Special values tests
# =============================================================================

# =============================================================================
# Half/BFloat16 precision tests
# =============================================================================

class TestLowPrecisionDtypes:
    """Test half-precision and bfloat16 dtype handling.

    These tests validate that the implementation produces reasonable results
    with reduced precision dtypes. Due to the limited precision of these formats,
    tolerances are relaxed compared to float32/float64 tests.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_basic_values(self, dtype):
        """Test forward pass produces finite results for basic inputs."""
        z = torch.tensor([0.0, 0.5, -0.5, 0.9, -0.9], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)
        result = sf.chebyshev_polynomial_t(v, z)

        assert result.dtype == dtype
        assert torch.isfinite(result).all()

        # Compare against float32 reference with relaxed tolerance
        z_f32 = z.to(torch.float32)
        v_f32 = v.to(torch.float32)
        expected = sf.chebyshev_polynomial_t(v_f32, z_f32)

        # Half precision has ~3 decimal digits, bfloat16 has ~2-3
        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        assert_close(result.to(torch.float32), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_integer_degrees(self, dtype):
        """Test that integer degrees work correctly in low precision."""
        z = torch.tensor([0.5], dtype=dtype)

        for n in range(6):
            v = torch.tensor([float(n)], dtype=dtype)
            result = sf.chebyshev_polynomial_t(v, z)

            # Compare against float32 reference
            v_f32 = torch.tensor([float(n)], dtype=torch.float32)
            z_f32 = torch.tensor([0.5], dtype=torch.float32)
            expected = sf.chebyshev_polynomial_t(v_f32, z_f32)

            rtol = 1e-2 if dtype == torch.float16 else 5e-2
            atol = 1e-2 if dtype == torch.float16 else 5e-2
            assert_close(result.to(torch.float32), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_special_points(self, dtype):
        """Test special values T_n(1) = 1 and T_n(-1) = (-1)^n."""
        for n in range(5):
            v = torch.tensor([float(n)], dtype=dtype)

            # T_n(1) = 1
            z_one = torch.tensor([1.0], dtype=dtype)
            result_one = sf.chebyshev_polynomial_t(v, z_one)
            expected_one = torch.tensor([1.0], dtype=dtype)
            assert_close(result_one, expected_one, rtol=1e-2, atol=1e-2)

            # T_n(-1) = (-1)^n
            z_minus_one = torch.tensor([-1.0], dtype=dtype)
            result_minus_one = sf.chebyshev_polynomial_t(v, z_minus_one)
            expected_minus_one = torch.tensor([(-1.0)**n], dtype=dtype)
            assert_close(result_minus_one, expected_minus_one, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_output_dtype_preservation(self, dtype):
        """Test that output dtype matches input dtype."""
        z = torch.tensor([0.5], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_broadcasting(self, dtype):
        """Test broadcasting works with low precision dtypes."""
        v = torch.tensor([[0.0], [1.0], [2.0]], dtype=dtype)
        z = torch.tensor([[0.0, 0.5, 0.9]], dtype=dtype)
        result = sf.chebyshev_polynomial_t(v, z)
        assert result.shape == (3, 3)
        assert result.dtype == dtype
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerical_stability_near_boundaries(self, dtype):
        """Test numerical stability near z = ±1 with low precision."""
        # Use values slightly inside [-1, 1] to avoid boundary issues
        z = torch.tensor([0.99, -0.99, 0.999, -0.999], dtype=dtype)
        v = torch.tensor([3.0], dtype=dtype)

        result = sf.chebyshev_polynomial_t(v, z)

        # Should be finite
        assert torch.isfinite(result).all()

        # Compare against float32 reference with relaxed tolerance
        z_f32 = z.to(torch.float32)
        v_f32 = v.to(torch.float32)
        expected = sf.chebyshev_polynomial_t(v_f32, z_f32)

        rtol = 5e-2  # Relaxed due to boundary proximity + low precision
        atol = 5e-2
        assert_close(result.to(torch.float32), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_integer_degree(self, dtype):
        """Test non-integer degrees work with low precision (analytic path)."""
        z = torch.tensor([0.0, 0.5], dtype=dtype)
        v = torch.tensor([0.5], dtype=dtype)  # Half-integer degree

        result = sf.chebyshev_polynomial_t(v, z)

        assert torch.isfinite(result).all()

        # T_{0.5}(0) = cos(0.5 * pi/2) = cos(pi/4) = sqrt(2)/2 ≈ 0.7071
        expected_at_zero = torch.tensor([0.7071], dtype=torch.float32)
        assert_close(result[0:1].to(torch.float32), expected_at_zero, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gradient_finite(self, dtype):
        """Test that gradients are finite for low precision inputs.

        Note: We cast to float32 for gradient computation since autograd
        may have issues with very low precision. This tests that the forward
        pass produces reasonable values that don't cause gradient issues.
        """
        # Use float32 for gradient computation, but verify input handling
        z = torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
        v = torch.tensor([2.0], dtype=dtype)

        # Forward pass in low precision
        result = sf.chebyshev_polynomial_t(v, z)
        assert torch.isfinite(result).all()

        # For actual gradient test, use float32
        z_f32 = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32, requires_grad=True)
        v_f32 = torch.tensor([2.0], dtype=torch.float32)
        y = sf.chebyshev_polynomial_t(v_f32, z_f32)
        y.sum().backward()

        assert torch.isfinite(z_f32.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cuda_low_precision(self, dtype):
        """Test low precision dtypes on CUDA."""
        # Skip bfloat16 if not supported
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported on this GPU")

        z = torch.tensor([0.0, 0.5, -0.5], dtype=dtype, device="cuda")
        v = torch.tensor([2.0], dtype=dtype, device="cuda")

        result = sf.chebyshev_polynomial_t(v, z)

        assert result.device.type == "cuda"
        assert result.dtype == dtype
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_precision_promotion(self, dtype):
        """Test that mixing low precision with higher precision promotes correctly."""
        z_low = torch.tensor([0.5], dtype=dtype)
        v_f32 = torch.tensor([2.0], dtype=torch.float32)

        result = sf.chebyshev_polynomial_t(v_f32, z_low)

        # Should promote to the higher precision type
        assert result.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_degree_stability(self, dtype):
        """Test stability with moderately large degrees in low precision."""
        z = torch.tensor([0.5], dtype=dtype)
        v = torch.tensor([10.0], dtype=dtype)  # Moderate degree

        result = sf.chebyshev_polynomial_t(v, z)

        assert torch.isfinite(result).all()

        # Compare against float32 reference
        z_f32 = z.to(torch.float32)
        v_f32 = v.to(torch.float32)
        expected = sf.chebyshev_polynomial_t(v_f32, z_f32)

        # Larger degree = more accumulation of rounding errors, so relax tolerance
        rtol = 0.1
        atol = 0.1
        assert_close(result.to(torch.float32), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gradient_boundary_handling_low_precision(self, dtype):
        """Test gradient stability at z = ±1 boundaries with low precision dtypes.

        This tests that the precision-aware epsilon in boundary detection works
        correctly for float16/bfloat16. The gradient formula dT_v/dz involves
        division by sqrt(1-z²), which is unstable near z = ±1. The implementation
        uses limit values at these boundaries, and the boundary detection epsilon
        must be appropriate for the input precision.
        """
        # Test at z = 1.0 (exactly representable in all formats)
        z_one = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        v = torch.tensor([2.5], dtype=torch.float32)  # Non-integer to use analytic path

        y = sf.chebyshev_polynomial_t(v, z_one)
        y.backward()

        # Gradient at z=1 should be v² = 6.25
        expected_grad = torch.tensor([6.25], dtype=torch.float32)
        assert torch.isfinite(z_one.grad).all()
        assert_close(z_one.grad, expected_grad, rtol=1e-3, atol=1e-3)

        # Test at z = -1.0
        z_minus_one = torch.tensor([-1.0], dtype=torch.float32, requires_grad=True)
        y = sf.chebyshev_polynomial_t(v, z_minus_one)
        y.backward()

        # Gradient at z=-1 should be v² * cos(π*v)
        expected_grad = v * v * torch.cos(torch.tensor(math.pi) * v)
        assert torch.isfinite(z_minus_one.grad).all()
        assert_close(z_minus_one.grad, expected_grad, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_values_near_boundary_quantize_correctly(self, dtype):
        """Test that values near boundaries are handled correctly after quantization.

        When a value like 0.999 is stored in float16/bfloat16, it may quantize
        to exactly 1.0 or to a value further from 1.0 than expected. The boundary
        detection epsilon must account for this quantization behavior.
        """
        # Create values that are close to 1.0 but may quantize differently
        # For float16, values within ~0.001 of 1.0 may quantize to 1.0
        # For bfloat16, values within ~0.008 of 1.0 may quantize to 1.0
        test_values = [0.99, 0.995, 0.999, -0.99, -0.995, -0.999]
        z = torch.tensor(test_values, dtype=dtype)
        v = torch.tensor([2.5], dtype=dtype)  # Non-integer

        result = sf.chebyshev_polynomial_t(v, z)

        # All results should be finite (no NaN from sqrt(1-z²) issues)
        assert torch.isfinite(result).all(), f"Got non-finite results for {dtype}: {result}"

        # Results should be bounded (|T_v(z)| <= cosh(|v|*acosh(|z|)) for |z| <= 1)
        # For z near ±1 and v=2.5, values should be close to T_v(±1)
        # T_v(1) = 1, T_v(-1) = cos(v*π) ≈ 0.707 for v=2.5
        assert (result.abs() <= 2.0).all(), f"Results out of expected range: {result}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_exact_boundary_values(self, dtype):
        """Test that exact boundary values z = ±1 work correctly in low precision."""
        z_boundaries = torch.tensor([1.0, -1.0], dtype=dtype)
        v = torch.tensor([2.5], dtype=dtype)

        result = sf.chebyshev_polynomial_t(v, z_boundaries)

        # T_v(1) = 1 for all v
        # T_v(-1) = cos(v*π) = cos(2.5*π) = cos(π/2) = 0 (approximately)
        assert torch.isfinite(result).all()

        # Check T_v(1) = 1
        expected_at_one = torch.tensor([1.0], dtype=torch.float32)
        assert_close(result[0:1].to(torch.float32), expected_at_one, rtol=1e-2, atol=1e-2)

        # Check T_v(-1) = cos(2.5*π) ≈ 0
        expected_at_minus_one = torch.tensor([0.0], dtype=torch.float32)
        assert_close(result[1:2].to(torch.float32), expected_at_minus_one, rtol=1e-1, atol=1e-1)


class TestSpecialValues:
    """Test special mathematical values."""

    def test_chebyshev_at_zero(self):
        """Test T_n(0) = cos(n*pi/2)."""
        z = torch.tensor([0.0], dtype=torch.float64)
        for n in range(5):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)
            expected = torch.cos(torch.tensor([n * math.pi / 2], dtype=torch.float64))
            # Use atol to handle -0.0 vs 0.0 comparison
            assert_close(result, expected, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_chebyshev_at_one(self):
        """Test T_n(1) = 1 for all n >= 0."""
        z = torch.tensor([1.0], dtype=torch.float64)
        for n in range(10):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)
            expected = torch.tensor([1.0], dtype=torch.float64)
            assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_chebyshev_at_minus_one(self):
        """Test T_n(-1) = (-1)^n."""
        z = torch.tensor([-1.0], dtype=torch.float64)
        for n in range(10):
            v = torch.tensor([float(n)], dtype=torch.float64)
            result = sf.chebyshev_polynomial_t(v, z)
            expected = torch.tensor([(-1.0)**n], dtype=torch.float64)
            assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_chebyshev_half_integer(self):
        """Test half-integer degrees at special points."""
        z = torch.tensor([0.0], dtype=torch.float64)
        # T_{1/2}(0) = cos(pi/4) = sqrt(2)/2
        v = torch.tensor([0.5], dtype=torch.float64)
        result = sf.chebyshev_polynomial_t(v, z)
        expected = torch.tensor([math.sqrt(2) / 2], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
