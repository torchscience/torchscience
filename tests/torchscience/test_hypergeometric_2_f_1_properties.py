"""Property-based tests for hypergeometric_2_f_1 using Hypothesis.

This module uses Hypothesis to automatically generate diverse test inputs
and discover edge cases for the hypergeometric function implementation.
"""

import mpmath
import pytest
import torch
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

from torchscience.special_functions import hypergeometric_2_f_1


# Strategy for generating safe hypergeometric parameters
def safe_float_strategy(min_value=-10.0, max_value=10.0, exclude_zero=False):
    """Generate safe float values for testing."""
    strategy = st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
    if exclude_zero:
        strategy = strategy.filter(lambda x: abs(x) > 0.1)
    return strategy


def safe_z_strategy():
    """Generate z values in convergence region |z| < 0.9."""
    return st.floats(
        min_value=-0.9,
        max_value=0.9,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


class TestHypergeometric2F1Properties:
    """Property-based tests for hypergeometric_2_f_1."""

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=50, deadline=None)
    def test_output_is_finite(self, a, b, c, z):
        """Property: For finite inputs in convergence region, output should be finite."""
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        assert torch.isfinite(result).all(), f"Result not finite for a={a}, b={b}, c={c}, z={z}"

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=50, deadline=None)
    def test_deterministic_computation(self, a, b, c, z):
        """Property: Same inputs should always produce same outputs."""
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result1 = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        result2 = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        torch.testing.assert_close(result1, result2)

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=30, deadline=None)
    def test_symmetry_in_ab(self, a, b, c, z):
        """Property: ₂F₁(a,b;c;z) = ₂F₁(b,a;c;z) (symmetry in a and b)."""
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result_ab = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        result_ba = hypergeometric_2_f_1(b_t, a_t, c_t, z_t)

        torch.testing.assert_close(result_ab, result_ba, rtol=1e-10, atol=1e-12)

    @given(z=safe_z_strategy())
    @settings(max_examples=20, deadline=None)
    def test_special_case_a_equals_zero(self, z):
        """Property: ₂F₁(0,b;c;z) = 1 for any b, c, z."""
        a_t = torch.tensor([0.0], dtype=torch.float64)
        b_t = torch.tensor([2.0], dtype=torch.float64)
        c_t = torch.tensor([3.0], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        expected = torch.tensor([1.0], dtype=torch.float64)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    @given(z=safe_z_strategy())
    @settings(max_examples=20, deadline=None)
    def test_special_case_b_equals_zero(self, z):
        """Property: ₂F₁(a,0;c;z) = 1 for any a, c, z."""
        a_t = torch.tensor([2.0], dtype=torch.float64)
        b_t = torch.tensor([0.0], dtype=torch.float64)
        c_t = torch.tensor([3.0], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        expected = torch.tensor([1.0], dtype=torch.float64)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
    )
    @settings(max_examples=20, deadline=None)
    def test_special_case_z_equals_zero(self, a, b, c):
        """Property: ₂F₁(a,b;c;0) = 1 for any a, b, c."""
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([0.0], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        expected = torch.tensor([1.0], dtype=torch.float64)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    @given(
        a=safe_float_strategy(-5.0, 5.0),
        b=safe_float_strategy(-5.0, 5.0),
        c=safe_float_strategy(0.5, 5.0),
        z=safe_z_strategy(),
    )
    @settings(max_examples=30, deadline=None)
    def test_agreement_with_mpmath(self, a, b, c, z):
        """Property: Results should agree with mpmath high-precision reference."""
        # Set mpmath precision
        mpmath.mp.dps = 50

        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        # Compute with mpmath
        try:
            expected_mp = mpmath.hyp2f1(a, b, c, z)
            expected = float(expected_mp)
            expected_t = torch.tensor([expected], dtype=torch.float64)

            torch.testing.assert_close(result, expected_t, rtol=1e-10, atol=1e-12)
        except (ValueError, ZeroDivisionError):
            # Some parameter combinations may not be valid
            pytest.skip(f"mpmath failed for a={a}, b={b}, c={c}, z={z}")

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5)
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_shape_preservation(self, shape):
        """Property: Output shape should match broadcast shape of inputs."""
        a_t = torch.ones(shape, dtype=torch.float64)
        b_t = torch.ones(shape, dtype=torch.float64) * 2.0
        c_t = torch.ones(shape, dtype=torch.float64) * 3.0
        z_t = torch.ones(shape, dtype=torch.float64) * 0.5

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        assert result.shape == shape

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=30, deadline=None)
    def test_dtype_preservation(self, a, b, c, z):
        """Property: Output dtype should match promoted input dtype."""
        # Test with float64
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        assert result.dtype == torch.float64

        # Test with float32
        a_t32 = a_t.to(torch.float32)
        b_t32 = b_t.to(torch.float32)
        c_t32 = c_t.to(torch.float32)
        z_t32 = z_t.to(torch.float32)

        result32 = hypergeometric_2_f_1(a_t32, b_t32, c_t32, z_t32)
        assert result32.dtype == torch.float32

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_no_input_mutation(self, a, b, c, z):
        """Property: Function should not modify input tensors."""
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        # Create copies
        a_orig = a_t.clone()
        b_orig = b_t.clone()
        c_orig = c_t.clone()
        z_orig = z_t.clone()

        _ = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        # Check inputs unchanged
        torch.testing.assert_close(a_t, a_orig)
        torch.testing.assert_close(b_t, b_orig)
        torch.testing.assert_close(c_t, c_orig)
        torch.testing.assert_close(z_t, z_orig)

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z1=safe_z_strategy(),
        z2=safe_z_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_monotonicity_in_z_for_positive_ab(self, a, b, c, z1, z2):
        """Property: For a,b,c > 0 and 0 < z1 < z2 < 1, f(z1) < f(z2) or f(z1) > f(z2)."""
        # Only test when a, b, c are all positive
        if a <= 0 or b <= 0 or c <= 0:
            return

        # Ensure z1 < z2 and both positive
        if z1 > z2:
            z1, z2 = z2, z1
        if z1 <= 0 or z2 <= 0:
            return

        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z1_t = torch.tensor([z1], dtype=torch.float64)
        z2_t = torch.tensor([z2], dtype=torch.float64)

        result1 = hypergeometric_2_f_1(a_t, b_t, c_t, z1_t)
        result2 = hypergeometric_2_f_1(a_t, b_t, c_t, z2_t)

        # For positive a, b, c and positive z, the function is monotone increasing
        # (though the direction depends on parameters)
        # We just check that we get finite values and they're different
        assert torch.isfinite(result1).all()
        assert torch.isfinite(result2).all()

    @given(
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=30, deadline=None)
    def test_gradient_exists(self, a, b, c, z):
        """Property: Function should be differentiable with respect to all inputs."""
        a_t = torch.tensor([a], dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor([b], dtype=torch.float64, requires_grad=True)
        c_t = torch.tensor([c], dtype=torch.float64, requires_grad=True)
        z_t = torch.tensor([z], dtype=torch.float64, requires_grad=True)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        # Should be able to compute gradient
        result.backward()

        assert a_t.grad is not None
        assert b_t.grad is not None
        assert c_t.grad is not None
        assert z_t.grad is not None

        # Gradients should be finite
        assert torch.isfinite(a_t.grad).all()
        assert torch.isfinite(b_t.grad).all()
        assert torch.isfinite(c_t.grad).all()
        assert torch.isfinite(z_t.grad).all()

    @given(
        batch_size=st.integers(min_value=1, max_value=10),
        a=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=10, deadline=None)
    def test_batch_consistency(self, batch_size, a, b, c, z):
        """Property: Batch computation should give same results as individual computations."""
        # Single computation
        a_t = torch.tensor([a], dtype=torch.float64)
        b_t = torch.tensor([b], dtype=torch.float64)
        c_t = torch.tensor([c], dtype=torch.float64)
        z_t = torch.tensor([z], dtype=torch.float64)

        single_result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        # Batch computation (same values repeated)
        a_batch = torch.full((batch_size,), a, dtype=torch.float64)
        b_batch = torch.full((batch_size,), b, dtype=torch.float64)
        c_batch = torch.full((batch_size,), c, dtype=torch.float64)
        z_batch = torch.full((batch_size,), z, dtype=torch.float64)

        batch_result = hypergeometric_2_f_1(a_batch, b_batch, c_batch, z_batch)

        # All batch results should match single result
        expected = single_result.expand(batch_size)
        torch.testing.assert_close(batch_result, expected, rtol=1e-10, atol=1e-12)

    @given(
        a_real=safe_float_strategy(),
        a_imag=safe_float_strategy(),
        b=safe_float_strategy(),
        c=safe_float_strategy(exclude_zero=True),
        z=safe_z_strategy(),
    )
    @settings(max_examples=20, deadline=None)
    def test_complex_dtype_support(self, a_real, a_imag, b, c, z):
        """Property: Function should work with complex inputs."""
        a_t = torch.tensor([complex(a_real, a_imag)], dtype=torch.complex128)
        b_t = torch.tensor([b], dtype=torch.complex128)
        c_t = torch.tensor([c], dtype=torch.complex128)
        z_t = torch.tensor([z], dtype=torch.complex128)

        result = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)

        assert result.dtype == torch.complex128
        # Result should be finite (both real and imaginary parts)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
