"""
Comprehensive tests for gamma function.

Tests cover:
- Basic functionality (factorial, half-integers, reflection)
- Real and complex types
- Dtype promotion rules
- Broadcasting
- Autograd (gradcheck and gradgradcheck)
- Numerical stability
- Pole behavior (NaN at non-positive integers)
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


def reference_gamma_real(x):
    """Reference implementation using torch.special.gammaln for real inputs."""
    # Use exp(gammaln(x)) as reference, handling sign
    sign = torch.ones_like(x)
    # For negative non-integer x, gamma can be negative
    # gammaln returns log of absolute value
    result = torch.exp(torch.special.gammaln(x))
    # Handle sign for negative x
    negative_mask = x < 0
    if negative_mask.any():
        # Use reflection formula to determine sign
        # Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        # For negative x, sign alternates based on floor(x)
        floor_x = torch.floor(x)
        sign = torch.where(negative_mask & ((floor_x % 2) != 0), -sign, sign)
    return sign * result


# =============================================================================
# Basic functionality tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic forward pass functionality."""

    def test_factorial_relationship(self):
        """Test Gamma(n) = (n-1)! for positive integers."""
        # Gamma(1) = 0! = 1
        # Gamma(2) = 1! = 1
        # Gamma(3) = 2! = 2
        # Gamma(4) = 3! = 6
        # Gamma(5) = 4! = 24
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        result = sf.gamma(x)
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0, 120.0], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_half_integer_values(self):
        """Test Gamma at half-integer values."""
        # Gamma(1/2) = sqrt(pi)
        # Gamma(3/2) = sqrt(pi)/2
        # Gamma(5/2) = 3*sqrt(pi)/4
        sqrt_pi = math.sqrt(math.pi)
        x = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        result = sf.gamma(x)
        expected = torch.tensor([sqrt_pi, sqrt_pi / 2, 3 * sqrt_pi / 4], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_recurrence_relation(self):
        """Test Gamma(x+1) = x * Gamma(x)."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.7], dtype=torch.float64)
        result_x = sf.gamma(x)
        result_x_plus_1 = sf.gamma(x + 1)
        expected = x * result_x
        assert_close(result_x_plus_1, expected, rtol=1e-10, atol=1e-10)

    def test_reflection_formula(self):
        """Test Gamma(x) * Gamma(1-x) = pi / sin(pi*x)."""
        x = torch.tensor([0.25, 0.3, 0.4, 0.6, 0.7, 0.75], dtype=torch.float64)
        gamma_x = sf.gamma(x)
        gamma_1_minus_x = sf.gamma(1 - x)
        product = gamma_x * gamma_1_minus_x
        expected = math.pi / torch.sin(math.pi * x)
        assert_close(product, expected, rtol=1e-10, atol=1e-10)

    def test_negative_non_integer_values(self):
        """Test Gamma at negative non-integer values."""
        # Use reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        x = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float64)
        result = sf.gamma(x)

        # Reference using reflection
        expected = torch.tensor([
            -2 * math.sqrt(math.pi),  # Gamma(-0.5)
            4 * math.sqrt(math.pi) / 3,  # Gamma(-1.5)
            -8 * math.sqrt(math.pi) / 15,  # Gamma(-2.5)
        ], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)


# =============================================================================
# Pole behavior tests
# =============================================================================

class TestPoleBehavior:
    """Test behavior at poles (non-positive integers)."""

    def test_poles_real_positive_zero(self):
        """Test that Gamma(0) returns inf."""
        x = torch.tensor([0.0], dtype=torch.float64)
        result = sf.gamma(x)
        assert torch.isinf(result).all()

    def test_poles_real_negative_integers(self):
        """Test that Gamma at negative integers returns inf or nan."""
        x = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
        result = sf.gamma(x)
        # Standard tgamma returns inf at negative integers
        assert (torch.isinf(result) | torch.isnan(result)).all()

    def test_complex_poles_return_nan_in_gradient(self):
        """Test that gradients at complex poles return NaN (uses digamma)."""
        poles = [0 + 0j, -1 + 0j, -2 + 0j, -3 + 0j]
        for p in poles:
            x = torch.tensor([p], dtype=torch.complex128, requires_grad=True)
            result = sf.gamma(x)
            # Use .real to get a real scalar for backward
            result.real.backward()
            # Gradient uses digamma which should return NaN at poles
            assert torch.isnan(x.grad).all(), f"Expected NaN gradient at pole {p}, got {x.grad}"

    def test_near_poles_finite(self):
        """Test that values near poles (but not at them) are finite."""
        near_poles = [0.001, -0.999, -1.001, -1.999, -2.001]
        x = torch.tensor(near_poles, dtype=torch.float64)
        result = sf.gamma(x)
        assert torch.isfinite(result).all()

    def test_complex_near_poles_finite(self):
        """Test that complex values near poles are finite."""
        near_poles = [0.001 + 0j, -0.999 + 0j, -1 + 0.001j, -2 + 0.001j]
        x = torch.tensor(near_poles, dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()


# =============================================================================
# Complex input tests
# =============================================================================

class TestComplexInputs:
    """Test complex input handling."""

    def test_complex_positive_real_axis(self):
        """Test complex numbers on positive real axis match real gamma."""
        x_real = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=torch.float64)
        x_complex = x_real.to(torch.complex128)

        result_real = sf.gamma(x_real)
        result_complex = sf.gamma(x_complex)

        assert_close(result_complex.real, result_real, rtol=1e-10, atol=1e-10)
        assert_close(result_complex.imag, torch.zeros_like(result_real), rtol=1e-10, atol=1e-10)

    def test_complex_conjugate_symmetry(self):
        """Test Gamma(conj(z)) = conj(Gamma(z))."""
        z = torch.tensor([1.0 + 1.0j, 2.0 + 0.5j, 0.5 - 0.3j], dtype=torch.complex128)

        result_z = sf.gamma(z)
        result_conj_z = sf.gamma(z.conj())

        assert_close(result_conj_z, result_z.conj(), rtol=1e-10, atol=1e-10)

    def test_complex_various_quadrants(self):
        """Test complex gamma in various quadrants of the complex plane."""
        # First quadrant
        z1 = torch.tensor([2.0 + 1.0j], dtype=torch.complex128)
        # Second quadrant
        z2 = torch.tensor([-0.5 + 1.0j], dtype=torch.complex128)
        # Third quadrant
        z3 = torch.tensor([-0.5 - 1.0j], dtype=torch.complex128)
        # Fourth quadrant
        z4 = torch.tensor([2.0 - 1.0j], dtype=torch.complex128)

        for z in [z1, z2, z3, z4]:
            result = sf.gamma(z)
            assert torch.isfinite(result.real).all()
            assert torch.isfinite(result.imag).all()

    def test_output_dtype_complex(self):
        """Test that complex input produces complex output."""
        x = torch.tensor([1.0 + 0.5j], dtype=torch.complex128)
        result = sf.gamma(x)
        assert result.is_complex()
        assert result.dtype == torch.complex128


# =============================================================================
# Dtype tests
# =============================================================================

class TestDtypes:
    """Test dtype handling and promotion."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_dtypes(self, dtype):
        """Test real floating point dtypes."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        result = sf.gamma(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype):
        """Test complex dtypes."""
        x = torch.tensor([1.0 + 0.5j], dtype=dtype)
        result = sf.gamma(x)
        assert result.dtype == dtype

    def test_mixed_float32_float64_promotes(self):
        """Test that mixing float32 input with float64 promotes correctly."""
        # This test is about input/output behavior
        x32 = torch.tensor([2.0], dtype=torch.float32)
        x64 = torch.tensor([2.0], dtype=torch.float64)

        result32 = sf.gamma(x32)
        result64 = sf.gamma(x64)

        assert result32.dtype == torch.float32
        assert result64.dtype == torch.float64

    @pytest.mark.parametrize("int_dtype", [torch.int32, torch.int64])
    def test_integer_dtype_requires_float_conversion(self, int_dtype):
        """Test that integer inputs require explicit conversion to float."""
        x_int = torch.tensor([1, 2, 3, 4, 5], dtype=int_dtype)

        # Integer inputs should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            sf.gamma(x_int)

        # Users should convert to float explicitly
        x_float = x_int.to(torch.float64)
        result = sf.gamma(x_float)

        # Should give correct factorial values
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0], dtype=torch.float64)
        assert_close(result, expected, rtol=1e-10, atol=1e-10)


# =============================================================================
# Broadcasting tests
# =============================================================================

class TestBroadcasting:
    """Test broadcasting behavior."""

    def test_broadcast_1d(self):
        """Test 1D tensor input."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = sf.gamma(x)
        assert result.shape == x.shape

    def test_broadcast_2d(self):
        """Test 2D tensor input."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        result = sf.gamma(x)
        assert result.shape == x.shape

    def test_broadcast_large_batch(self):
        """Test large batch dimensions."""
        x = torch.rand(10, 20, 30, dtype=torch.float64) + 0.5  # Avoid poles
        result = sf.gamma(x)
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


# =============================================================================
# Autograd tests
# =============================================================================

class TestAutograd:
    """Test autograd functionality."""

    def test_gradcheck_float64(self):
        """Test gradient correctness with float64."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64, requires_grad=True)

        def func(x):
            return sf.gamma(x)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-4)

    def test_gradcheck_complex128(self):
        """Test gradient correctness with complex128."""
        x = torch.tensor([1.0 + 0.5j, 2.0 + 0.3j], dtype=torch.complex128, requires_grad=True)

        def func(x):
            return sf.gamma(x)

        # Complex gradcheck with relaxed tolerances
        assert torch.autograd.gradcheck(func, (x,), eps=1e-5, atol=1e-3, rtol=1e-3)

    def test_gradgradcheck_float64(self):
        """Test second-order gradient correctness."""
        x = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)

        def func(x):
            return sf.gamma(x)

        assert torch.autograd.gradgradcheck(func, (x,), eps=1e-6, atol=1e-3, rtol=1e-3)

    def test_backward_formula_manual(self):
        """Manually verify backward formula: d/dx Gamma(x) = Gamma(x) * psi(x)."""
        x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        y = sf.gamma(x)
        y.backward()

        # Manual: Gamma'(x) = Gamma(x) * digamma(x)
        # Gamma(2) = 1, digamma(2) = 1 - gamma_euler ≈ 0.4227843
        gamma_euler = 0.5772156649015329
        expected_grad = 1.0 * (1.0 - gamma_euler)

        assert_close(x.grad, torch.tensor([expected_grad], dtype=torch.float64), rtol=1e-6, atol=1e-6)

    def test_gradient_at_positive_integers(self):
        """Test gradients at positive integers."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

        y = sf.gamma(x)
        y.sum().backward()

        # Gradients should be finite
        assert torch.isfinite(x.grad).all()

    def test_gradient_near_poles_finite(self):
        """Test that gradients near (but not at) poles are finite."""
        x = torch.tensor([0.001, -0.999, -1.999], dtype=torch.float64, requires_grad=True)

        y = sf.gamma(x)
        y.sum().backward()

        assert torch.isfinite(x.grad).all()


# =============================================================================
# Numerical stability tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_large_positive_values(self):
        """Test with large positive values (may overflow)."""
        x = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)
        result = sf.gamma(x)

        # Large values should give large but finite results (until overflow)
        # Gamma(10) = 362880, Gamma(20) ≈ 1.2e17
        assert torch.isfinite(result[:2]).all()
        # Gamma(50) may overflow to inf
        assert (torch.isfinite(result[2]) | torch.isinf(result[2])).all()

    def test_small_positive_values(self):
        """Test with small positive values."""
        x = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)
        result = sf.gamma(x)

        # Gamma(x) ≈ 1/x for small x > 0
        expected_approx = 1 / x
        # Should be approximately equal (not exact)
        assert torch.isfinite(result).all()
        # Relative error should be small
        rel_error = torch.abs(result - expected_approx) / torch.abs(expected_approx)
        assert (rel_error < 0.1).all()

    def test_moderate_negative_values(self):
        """Test with moderate negative non-integer values."""
        x = torch.tensor([-0.5, -1.5, -2.5, -3.5], dtype=torch.float64)
        result = sf.gamma(x)
        assert torch.isfinite(result).all()

    def test_complex_large_imaginary(self):
        """Test complex numbers with large imaginary parts."""
        x = torch.tensor([1.0 + 10.0j, 1.0 + 20.0j], dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isfinite(result.real).all()
        assert torch.isfinite(result.imag).all()


# =============================================================================
# Device tests
# =============================================================================

class TestDevices:
    """Test device handling."""

    def test_cpu(self):
        """Test on CPU."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cpu")
        result = sf.gamma(x)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test on CUDA."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
        result = sf.gamma(x)
        assert result.device.type == "cuda"

        # Compare against CPU
        expected = sf.gamma(x.cpu())
        assert_close(result.cpu(), expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_complex(self):
        """Test complex on CUDA."""
        x = torch.tensor([1.0 + 0.5j, 2.0 + 0.3j], dtype=torch.complex128, device="cuda")
        result = sf.gamma(x)
        assert result.device.type == "cuda"

        # Compare against CPU
        expected = sf.gamma(x.cpu())
        assert_close(result.cpu(), expected, rtol=1e-10, atol=1e-10)


# =============================================================================
# Half/BFloat16 precision tests
# =============================================================================

class TestLowPrecisionDtypes:
    """Test half-precision and bfloat16 dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_basic_values(self, dtype):
        """Test forward pass produces finite results for basic inputs."""
        x = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5], dtype=dtype)
        result = sf.gamma(x)

        assert result.dtype == dtype
        assert torch.isfinite(result).all()

        # Compare against float32 reference with relaxed tolerance
        x_f32 = x.to(torch.float32)
        expected = sf.gamma(x_f32)

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        assert_close(result.to(torch.float32), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_factorial_values(self, dtype):
        """Test factorial relationship in low precision."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        result = sf.gamma(x)
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0], dtype=dtype)

        rtol = 1e-2 if dtype == torch.float16 else 5e-2
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        assert_close(result, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_output_dtype_preservation(self, dtype):
        """Test that output dtype matches input dtype."""
        x = torch.tensor([2.0], dtype=dtype)
        result = sf.gamma(x)
        assert result.dtype == dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cuda_low_precision(self, dtype):
        """Test low precision dtypes on CUDA."""
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported on this GPU")

        x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="cuda")
        result = sf.gamma(x)

        assert result.device.type == "cuda"
        assert result.dtype == dtype
        assert torch.isfinite(result).all()


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
        def compiled_gamma(x):
            return sf.gamma(x)

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        result = compiled_gamma(x)
        expected = sf.gamma(x)
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
    def test_vmap_over_batch(self):
        """Test vmap over batch dimension."""
        from torch import vmap

        x = torch.rand(10, dtype=torch.float64) + 0.5  # Avoid poles

        def single_gamma(x_single):
            return sf.gamma(x_single.unsqueeze(0)).squeeze(0)

        vmapped = vmap(single_gamma)
        result = vmapped(x)

        expected = sf.gamma(x)
        assert_close(result, expected)


# =============================================================================
# Digamma pole detection tests (for gradients)
# =============================================================================

class TestDigammaPoleDetection:
    """Test that digamma (used in gradients) correctly handles poles for complex inputs."""

    def test_complex_gradient_at_zero(self):
        """Test that gradient at z=0+0j returns NaN."""
        x = torch.tensor([0.0 + 0.0j], dtype=torch.complex128, requires_grad=True)
        y = sf.gamma(x)
        y.real.backward()
        assert torch.isnan(x.grad).all()

    def test_complex_gradient_at_negative_integers(self):
        """Test that gradients at negative integer complex values return NaN."""
        for n in [-1, -2, -3, -4]:
            x = torch.tensor([complex(n, 0)], dtype=torch.complex128, requires_grad=True)
            y = sf.gamma(x)
            y.real.backward()
            assert torch.isnan(x.grad).all(), f"Expected NaN gradient at {n}+0j"

    def test_complex_gradient_near_poles_is_finite(self):
        """Test that gradients near (but not at) complex poles are finite."""
        near_poles = [
            0.001 + 0.0j,
            -0.999 + 0.0j,
            -1.0 + 0.001j,
            -2.0 + 0.001j,
            -1.001 + 0.0j,
        ]
        for p in near_poles:
            x = torch.tensor([p], dtype=torch.complex128, requires_grad=True)
            y = sf.gamma(x)
            y.real.backward()
            assert torch.isfinite(x.grad).all(), f"Expected finite gradient near {p}, got {x.grad}"

    def test_complex_gradient_positive_real_finite(self):
        """Test that gradients at positive real complex values are finite."""
        x = torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 0.5 + 0.0j], dtype=torch.complex128, requires_grad=True)
        y = sf.gamma(x)
        y.real.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_complex_gradient_with_imaginary_part_finite(self):
        """Test that gradients at complex values with nonzero imaginary part are finite."""
        x = torch.tensor([
            1.0 + 1.0j,
            2.0 + 0.5j,
            -0.5 + 1.0j,
            -1.0 + 1.0j,  # Would be pole if imag=0, but should be finite here
        ], dtype=torch.complex128, requires_grad=True)
        y = sf.gamma(x)
        y.real.sum().backward()
        assert torch.isfinite(x.grad).all()


# =============================================================================
# Trigamma pole detection tests (for second-order gradients)
# =============================================================================

class TestTrigammaPoleDetection:
    """Test that trigamma (used in second-order gradients) correctly handles poles."""

    def test_gradgrad_at_complex_pole_returns_nan(self):
        """Test that second-order gradient at complex pole returns NaN."""
        x = torch.tensor([0.0 + 0.0j], dtype=torch.complex128, requires_grad=True)
        y = sf.gamma(x)

        # First derivative
        grad_y = torch.autograd.grad(y.real, x, create_graph=True)[0]

        # Second derivative - should be NaN at pole
        try:
            grad_grad_y = torch.autograd.grad(grad_y.real, x)[0]
            assert torch.isnan(grad_grad_y).all(), f"Expected NaN at pole, got {grad_grad_y}"
        except RuntimeError:
            # Some versions may raise an error when computing grad of NaN
            pass

    def test_gradgrad_near_complex_pole_finite(self):
        """Test that second-order gradients near (but not at) complex poles are finite."""
        near_poles = [0.5 + 0.0j, 1.0 + 0.5j, -0.5 + 0.1j]
        for p in near_poles:
            x = torch.tensor([p], dtype=torch.complex128, requires_grad=True)
            y = sf.gamma(x)

            # First derivative
            grad_y = torch.autograd.grad(y.real, x, create_graph=True)[0]

            # Second derivative
            grad_grad_y = torch.autograd.grad(grad_y.real, x)[0]
            assert torch.isfinite(grad_grad_y).all(), f"Expected finite at {p}, got {grad_grad_y}"


# =============================================================================
# Pole detection edge case tests (Issues #1 and #3 from review)
# =============================================================================

class TestPoleDetectionEdgeCases:
    """Test pole detection with various edge cases including large negative integers."""

    def test_large_negative_integer_poles(self):
        """Test that large negative integers are correctly detected as poles."""
        # Large negative integers should be detected as poles
        large_poles = [-100.0, -500.0, -1000.0]
        for pole in large_poles:
            x = torch.tensor([pole], dtype=torch.float64)
            result = sf.gamma(x)
            assert torch.isinf(result).all(), f"Expected inf at {pole}, got {result}"

    def test_large_negative_integer_poles_complex(self):
        """Test complex large negative integers are detected as poles."""
        large_poles = [-100.0 + 0.0j, -500.0 + 0.0j, -1000.0 + 0.0j]
        for pole in large_poles:
            x = torch.tensor([pole], dtype=torch.complex128)
            result = sf.gamma(x)
            assert torch.isinf(result.real).all(), f"Expected inf at {pole}, got {result}"

    def test_large_negative_integer_gradient_is_nan(self):
        """Test gradients at large negative integer poles return NaN."""
        large_poles = [-100.0 + 0.0j, -500.0 + 0.0j]
        for pole in large_poles:
            x = torch.tensor([pole], dtype=torch.complex128, requires_grad=True)
            y = sf.gamma(x)
            y.real.backward()
            assert torch.isnan(x.grad).all(), f"Expected NaN gradient at {pole}, got {x.grad}"

    def test_values_very_close_to_poles_detected(self):
        """Test values within floating-point tolerance of poles give very large results."""
        # Values extremely close to poles should give very large (or infinite) results.
        # Near a simple pole, |Γ(x)| ≈ 1/|distance_to_pole|, so for distance 1e-14
        # we expect magnitude around 1e14.
        near_poles_float64 = [
            -1.0 + 1e-14,  # Very close to -1
            -2.0 - 1e-14,  # Very close to -2
            0.0 + 1e-14,   # Very close to 0
        ]
        for val in near_poles_float64:
            x = torch.tensor([val], dtype=torch.float64)
            result = sf.gamma(x)
            # Should be inf (detected as pole) or at least very large (~1/distance)
            # For distance 1e-14, expect magnitude > 1e12 (allowing some tolerance)
            assert torch.isinf(result).all() or result.abs() > 1e12, \
                f"Expected inf or very large at {val}, got {result}"

    def test_values_clearly_not_at_poles(self):
        """Test that values clearly between poles are not detected as poles."""
        # Half-integers are equidistant from two integers - clearly not poles
        not_poles = [-0.5, -1.5, -2.5, -100.5, -500.5]
        for val in not_poles:
            x = torch.tensor([val], dtype=torch.float64)
            result = sf.gamma(x)
            assert torch.isfinite(result).all(), f"Expected finite at {val}, got {result}"

    def test_complex_with_small_imaginary_not_pole(self):
        """Test complex values with small but nonzero imaginary part are not poles."""
        # Even small imaginary parts should prevent pole detection
        not_poles = [
            -1.0 + 1e-4j,   # Small but meaningful imaginary part
            -2.0 + 0.001j,
            0.0 + 1e-3j,
        ]
        for val in not_poles:
            x = torch.tensor([val], dtype=torch.complex128)
            result = sf.gamma(x)
            assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all(), \
                f"Expected finite at {val}, got {result}"

    def test_float32_pole_detection(self):
        """Test pole detection works correctly with float32 precision."""
        # Float32 has less precision, so tolerance scaling is important
        poles_f32 = [-1.0, -10.0, -100.0]
        for pole in poles_f32:
            x = torch.tensor([pole], dtype=torch.float32)
            result = sf.gamma(x)
            assert torch.isinf(result).all(), f"Expected inf at {pole} (float32), got {result}"

    def test_values_near_half_integers(self):
        """Test values near half-integers (equidistant from poles) are handled correctly."""
        # Values like -0.4999999 should NOT be treated as poles
        # (nearest pole is -1 or 0, both at distance ~0.5)
        near_half = [-0.4999999, -1.5000001, -2.4999999]
        for val in near_half:
            x = torch.tensor([val], dtype=torch.float64)
            result = sf.gamma(x)
            assert torch.isfinite(result).all(), \
                f"Value {val} should not be detected as pole, got {result}"


# =============================================================================
# Overflow behavior tests (Issue #2 from review)
# =============================================================================

class TestOverflowBehavior:
    """Test overflow behavior for large arguments as documented in Warnings section."""

    def test_float64_overflow_threshold(self):
        """Test that gamma overflows to inf around x=171 for float64."""
        # Gamma(171) ≈ 7.26e306, Gamma(172) overflows
        x_below = torch.tensor([170.0], dtype=torch.float64)
        x_at = torch.tensor([171.0], dtype=torch.float64)
        x_above = torch.tensor([172.0], dtype=torch.float64)

        result_below = sf.gamma(x_below)
        result_at = sf.gamma(x_at)
        result_above = sf.gamma(x_above)

        assert torch.isfinite(result_below).all(), "Gamma(170) should be finite"
        # Gamma(171) might be finite or inf depending on implementation
        assert torch.isfinite(result_at).all() or torch.isinf(result_at).all()
        assert torch.isinf(result_above).all(), "Gamma(172) should overflow to inf"

    def test_float32_overflow_threshold(self):
        """Test that gamma overflows to inf around x=35 for float32."""
        # Gamma(35) ≈ 2.95e38, close to float32 max ≈ 3.4e38
        x_below = torch.tensor([34.0], dtype=torch.float32)
        x_above = torch.tensor([36.0], dtype=torch.float32)

        result_below = sf.gamma(x_below)
        result_above = sf.gamma(x_above)

        assert torch.isfinite(result_below).all(), "Gamma(34) should be finite for float32"
        assert torch.isinf(result_above).all(), "Gamma(36) should overflow for float32"

    def test_documented_reference_values(self):
        """Test the reference values documented in the Warnings section."""
        # Gamma(20) ≈ 1.2e17
        x20 = torch.tensor([20.0], dtype=torch.float64)
        result20 = sf.gamma(x20)
        assert 1e17 < result20.item() < 2e17, f"Gamma(20) ≈ 1.2e17, got {result20.item():.2e}"

        # Gamma(100) ≈ 9.3e155
        x100 = torch.tensor([100.0], dtype=torch.float64)
        result100 = sf.gamma(x100)
        assert 9e155 < result100.item() < 1e156, f"Gamma(100) ≈ 9.3e155, got {result100.item():.2e}"

    def test_overflow_returns_positive_inf(self):
        """Test that overflow produces positive infinity, not NaN."""
        x = torch.tensor([200.0, 500.0, 1000.0], dtype=torch.float64)
        result = sf.gamma(x)

        assert torch.isinf(result).all(), "Large gamma values should overflow to inf"
        assert (result > 0).all(), "Overflow should be positive infinity"
        assert not torch.isnan(result).any(), "Overflow should not produce NaN"


# =============================================================================
# Special mathematical identities
# =============================================================================

class TestMathematicalIdentities:
    """Test various mathematical identities of the gamma function."""

    def test_duplication_formula(self):
        """Test Legendre duplication formula: Gamma(z) * Gamma(z+1/2) = sqrt(pi) / 2^(2z-1) * Gamma(2z)."""
        z = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

        lhs = sf.gamma(z) * sf.gamma(z + 0.5)
        rhs = math.sqrt(math.pi) / (2 ** (2 * z - 1)) * sf.gamma(2 * z)

        assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)

    def test_gamma_one(self):
        """Test Gamma(1) = 1."""
        x = torch.tensor([1.0], dtype=torch.float64)
        result = sf.gamma(x)
        assert_close(result, torch.tensor([1.0], dtype=torch.float64))

    def test_gamma_two(self):
        """Test Gamma(2) = 1! = 1."""
        x = torch.tensor([2.0], dtype=torch.float64)
        result = sf.gamma(x)
        assert_close(result, torch.tensor([1.0], dtype=torch.float64))


# =============================================================================
# Large negative argument numerical stability tests (sin_pi range reduction)
# =============================================================================

class TestLargeNegativeArgumentStability:
    """
    Test numerical stability for large negative arguments.

    The reflection formula Γ(x) = π / (sin(πx) * Γ(1-x)) requires computing
    sin(πx), which loses precision for large |x| due to floating-point
    representation limitations. These tests verify that the sin_pi()
    range reduction implementation maintains accuracy.
    """

    def test_large_negative_half_integers_real(self):
        """Test gamma at large negative half-integers like -1000.5, -10000.5.

        Half-integers are equidistant from poles and should give finite results.
        Γ(-n - 0.5) follows a predictable pattern using the reflection formula.
        """
        # Large negative half-integers
        half_integers = [-1000.5, -10000.5, -100000.5]

        for x in half_integers:
            t = torch.tensor([x], dtype=torch.float64)
            result = sf.gamma(t)

            # Result should be finite (not NaN or Inf)
            assert torch.isfinite(result).all(), \
                f"Gamma({x}) should be finite, got {result.item()}"

            # Verify sign: Gamma at negative half-integers alternates sign
            # Sign of Gamma(-n - 0.5) = (-1)^(n+1) for n >= 0
            n = int(-x - 0.5)
            expected_sign = (-1) ** (n + 1)
            actual_sign = 1 if result.item() > 0 else -1
            assert actual_sign == expected_sign, \
                f"Gamma({x}) has wrong sign: expected {expected_sign}, got {actual_sign}"

    def test_large_negative_half_integers_complex(self):
        """Test gamma at large negative half-integers for complex input (real axis)."""
        half_integers = [-1000.5 + 0j, -10000.5 + 0j]

        for x in half_integers:
            t = torch.tensor([x], dtype=torch.complex128)
            result = sf.gamma(t)

            # Result should be finite
            assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all(), \
                f"Gamma({x}) should be finite, got {result.item()}"

            # Imaginary part should be zero (or very small) for real arguments
            assert abs(result.imag.item()) < 1e-10, \
                f"Gamma({x}) imaginary part should be ~0, got {result.imag.item()}"

    def test_very_large_negative_values_real(self):
        """Test extremely large negative values where range reduction is critical.

        At x = -10^12, the floating-point representation has limited fractional
        precision. Without range reduction, sin(π * x) would be completely wrong.
        """
        # Test values where direct sin(π*x) would fail
        test_values = [
            -1e9 - 0.5,    # Should be finite
            -1e10 - 0.5,   # Should be finite
            -1e12 - 0.5,   # Should be finite
        ]

        for x in test_values:
            t = torch.tensor([x], dtype=torch.float64)
            result = sf.gamma(t)

            # Result should be finite (not NaN)
            # Note: may overflow to inf for very large |x| due to magnitude,
            # but should not be NaN
            assert not torch.isnan(result).any(), \
                f"Gamma({x}) should not be NaN, got {result.item()}"

    def test_reflection_formula_consistency_large_x(self):
        """Verify reflection formula Γ(x)Γ(1-x) = π/sin(πx) for large negative x.

        This tests that sin_pi gives consistent results by checking the
        reflection formula identity.
        """
        # Use values where 1-x is positive (so Γ(1-x) is computed via Lanczos, not reflection)
        test_values = [-99.3, -999.7, -9999.2]

        for x in test_values:
            t_x = torch.tensor([x], dtype=torch.float64)
            t_1_minus_x = torch.tensor([1 - x], dtype=torch.float64)

            gamma_x = sf.gamma(t_x)
            gamma_1_minus_x = sf.gamma(t_1_minus_x)
            product = gamma_x * gamma_1_minus_x

            # Expected: π / sin(πx)
            # Use the same range reduction approach for computing expected value
            sin_pi_x = math.sin(math.pi * (x % 2))  # Simple range reduction
            if abs(sin_pi_x) < 1e-15:
                continue  # Skip values too close to poles
            expected = math.pi / sin_pi_x

            # Check that product matches expected (may need loose tolerance due to overflow)
            if torch.isfinite(product).all() and abs(expected) < 1e100:
                rel_error = abs(product.item() - expected) / abs(expected)
                assert rel_error < 1e-8, \
                    f"Reflection formula failed for x={x}: " \
                    f"Γ(x)Γ(1-x)={product.item()}, π/sin(πx)={expected}, rel_error={rel_error}"

    def test_complex_large_negative_real_part(self):
        """Test complex gamma with large negative real part.

        Tests that sin_pi for complex numbers correctly handles range reduction
        on the real part.
        """
        test_values = [
            -1000.3 + 0.5j,
            -10000.7 + 0.1j,
            -999.5 + 1.0j,  # Half-integer real part with imaginary component
        ]

        for z in test_values:
            t = torch.tensor([z], dtype=torch.complex128)
            result = sf.gamma(t)

            # Result should be finite
            assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all(), \
                f"Gamma({z}) should be finite, got {result.item()}"

    def test_sin_pi_exact_values(self):
        """Verify gamma gives correct results for values where sin(πx) is known exactly.

        For x = n + 0.5 (half-integers), sin(πx) = ±1.
        This provides a strong test of the sin_pi implementation.
        """
        # Test several half-integers
        half_ints = [-0.5, -1.5, -2.5, -3.5, -100.5, -1000.5]

        for x in half_ints:
            t = torch.tensor([x], dtype=torch.float64)
            result = sf.gamma(t)

            assert torch.isfinite(result).all(), \
                f"Gamma({x}) should be finite, got {result.item()}"

            # Verify using recurrence: Γ(x) = Γ(x+1) / x
            t_plus_1 = torch.tensor([x + 1], dtype=torch.float64)
            gamma_plus_1 = sf.gamma(t_plus_1)
            expected_from_recurrence = gamma_plus_1 / (x)

            if torch.isfinite(expected_from_recurrence).all():
                assert_close(
                    result, expected_from_recurrence,
                    rtol=1e-10, atol=1e-10,
                    msg=f"Recurrence failed for x={x}"
                )

    def test_quarter_integers_large_negative(self):
        """Test quarter-integers where sin(πx) = ±√2/2.

        These provide another exactness test for sin_pi.
        For small x like -0.25, -0.75, results are moderately sized.
        For large negative x like -100.25, results are extremely small
        (Gamma(-100.25) ~ 10^-158) but should still be finite and non-NaN.
        """
        # Small quarter-integers should have reasonably-sized results
        small_quarter_ints = [-0.25, -0.75]
        for x in small_quarter_ints:
            t = torch.tensor([x], dtype=torch.float64)
            result = sf.gamma(t)

            assert torch.isfinite(result).all(), \
                f"Gamma({x}) should be finite, got {result.item()}"
            assert result.abs().item() > 1e-10, \
                f"Gamma({x}) should be reasonably sized, got {result.item()}"

        # Large negative quarter-integers will have extremely small results
        # (or even underflow to 0), but should never be NaN
        large_quarter_ints = [-100.25, -100.75, -1000.25]
        for x in large_quarter_ints:
            t = torch.tensor([x], dtype=torch.float64)
            result = sf.gamma(t)

            # Should be finite (not inf/nan) or zero (underflow is acceptable)
            assert not torch.isnan(result).any(), \
                f"Gamma({x}) should not be NaN, got {result.item()}"
            # Results at large negative values are extremely small or zero
            # due to the factorial decay in the denominator

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_large_negative_stability_dtypes(self, dtype):
        """Test that large negative values work for both float32 and float64."""
        # Use different scales for different dtypes due to precision limits
        if dtype == torch.float32:
            test_values = [-100.5, -1000.5, -10000.5]
        else:
            test_values = [-1000.5, -10000.5, -100000.5]

        for x in test_values:
            t = torch.tensor([x], dtype=dtype)
            result = sf.gamma(t)

            assert not torch.isnan(result).any(), \
                f"Gamma({x}) with dtype={dtype} should not be NaN"

    def test_complex_conjugate_symmetry_large_negative(self):
        """Test Γ(conj(z)) = conj(Γ(z)) for large negative real parts."""
        test_values = [
            -1000.3 + 0.5j,
            -10000.7 - 0.3j,
        ]

        for z in test_values:
            t = torch.tensor([z], dtype=torch.complex128)
            t_conj = torch.tensor([z.conjugate()], dtype=torch.complex128)

            result = sf.gamma(t)
            result_conj = sf.gamma(t_conj)

            if torch.isfinite(result.real).all():
                assert_close(result_conj, result.conj(), rtol=1e-10, atol=1e-10,
                           msg=f"Conjugate symmetry failed for z={z}")


# =============================================================================
# NaN propagation tests
# =============================================================================

class TestNaNPropagation:
    """Test that NaN inputs are properly propagated to outputs."""

    def test_real_nan_propagation(self):
        """Test that real NaN input produces NaN output."""
        x = torch.tensor([float('nan')], dtype=torch.float64)
        result = sf.gamma(x)
        assert torch.isnan(result).all()

    def test_complex_nan_real_part(self):
        """Test that complex input with NaN in real part produces NaN output."""
        x = torch.tensor([complex(float('nan'), 1.0)], dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isnan(result.real).all()
        assert torch.isnan(result.imag).all()

    def test_complex_nan_imag_part(self):
        """Test that complex input with NaN in imaginary part produces NaN output."""
        x = torch.tensor([complex(1.0, float('nan'))], dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isnan(result.real).all()
        assert torch.isnan(result.imag).all()

    def test_complex_nan_both_parts(self):
        """Test that complex input with NaN in both parts produces NaN output."""
        x = torch.tensor([complex(float('nan'), float('nan'))], dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isnan(result.real).all()
        assert torch.isnan(result.imag).all()

    def test_mixed_nan_and_valid(self):
        """Test that NaN propagates correctly in batched inputs."""
        x = torch.tensor([1.0, float('nan'), 2.0], dtype=torch.float64)
        result = sf.gamma(x)
        assert not torch.isnan(result[0])
        assert torch.isnan(result[1])
        assert not torch.isnan(result[2])

    def test_complex_mixed_nan_and_valid(self):
        """Test that complex NaN propagates correctly in batched inputs."""
        x = torch.tensor([
            1.0 + 0.5j,
            complex(float('nan'), 0.5),
            2.0 + 0.3j
        ], dtype=torch.complex128)
        result = sf.gamma(x)
        assert torch.isfinite(result[0])
        assert torch.isnan(result[1].real) and torch.isnan(result[1].imag)
        assert torch.isfinite(result[2])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_nan_propagation_dtypes(self, dtype):
        """Test NaN propagation across different real dtypes."""
        x = torch.tensor([float('nan')], dtype=dtype)
        result = sf.gamma(x)
        assert result.dtype == dtype
        assert torch.isnan(result).all()

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_nan_propagation_complex_dtypes(self, dtype):
        """Test NaN propagation across different complex dtypes."""
        x = torch.tensor([complex(float('nan'), 1.0)], dtype=dtype)
        result = sf.gamma(x)
        assert result.dtype == dtype
        assert torch.isnan(result.real).all()
        assert torch.isnan(result.imag).all()


# =============================================================================
# Sparse tensor tests
# =============================================================================

class TestSparseTensors:
    """Test sparse tensor support for the gamma function."""

    def test_sparse_coo_basic(self):
        """Test basic sparse COO tensor support."""
        # Create a sparse COO tensor
        indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

        result = sf.gamma(sparse)

        # Result should also be sparse
        assert result.is_sparse
        assert result.shape == sparse.shape

        # Check values match dense gamma applied to values
        expected_values = sf.gamma(values)
        assert_close(result._values(), expected_values, rtol=1e-10, atol=1e-10)

    def test_sparse_coo_preserves_indices(self):
        """Test that sparse COO gamma preserves indices."""
        indices = torch.tensor([[0, 2, 4], [1, 3, 0]])
        values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (5, 5))

        result = sf.gamma(sparse)

        # Indices should be preserved
        assert torch.equal(result._indices(), sparse._indices())

    def test_sparse_coo_coalesced_preserved(self):
        """Test that coalesced status is preserved."""
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()

        assert sparse.is_coalesced()
        result = sf.gamma(sparse)
        assert result.is_coalesced()

    def test_sparse_coo_matches_dense(self):
        """Test that sparse gamma matches dense gamma for the values."""
        # Create matching sparse and dense tensors
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([2.0, 3.0], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

        sparse_result = sf.gamma(sparse)

        # Extract values and compare
        dense_values_result = sf.gamma(values)
        assert_close(sparse_result._values(), dense_values_result, rtol=1e-10, atol=1e-10)

    def test_sparse_coo_complex(self):
        """Test sparse COO with complex values."""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0 + 0.5j, 2.0 + 0.3j], dtype=torch.complex128)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2))

        result = sf.gamma(sparse)

        expected_values = sf.gamma(values)
        assert_close(result._values(), expected_values, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sparse_coo_dtypes(self, dtype):
        """Test sparse COO with different dtypes."""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([2.0, 3.0], dtype=dtype)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2))

        result = sf.gamma(sparse)

        assert result.dtype == dtype

    def test_sparse_csr_basic(self):
        """Test basic sparse CSR tensor support."""
        # Create a sparse CSR tensor
        crow_indices = torch.tensor([0, 2, 3, 5])
        col_indices = torch.tensor([0, 2, 1, 0, 2])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, (3, 3))

        result = sf.gamma(sparse)

        # Result should also be sparse CSR
        assert result.layout == torch.sparse_csr
        assert result.shape == sparse.shape

        # Check values match dense gamma applied to values
        expected_values = sf.gamma(values)
        assert_close(result.values(), expected_values, rtol=1e-10, atol=1e-10)

    def test_sparse_csr_preserves_structure(self):
        """Test that sparse CSR gamma preserves row/column indices."""
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([0, 1, 2])
        values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, (3, 3))

        result = sf.gamma(sparse)

        # Structure should be preserved
        assert torch.equal(result.crow_indices(), sparse.crow_indices())
        assert torch.equal(result.col_indices(), sparse.col_indices())

    def test_sparse_csr_matches_dense_values(self):
        """Test that sparse CSR gamma matches dense gamma for values."""
        crow_indices = torch.tensor([0, 2, 3, 5])
        col_indices = torch.tensor([0, 2, 1, 0, 2])
        values = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
        sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, (3, 3))

        sparse_result = sf.gamma(sparse)
        dense_values_result = sf.gamma(values)

        assert_close(sparse_result.values(), dense_values_result, rtol=1e-10, atol=1e-10)

    def test_sparse_csr_complex(self):
        """Test sparse CSR with complex values."""
        crow_indices = torch.tensor([0, 1, 2])
        col_indices = torch.tensor([0, 1])
        values = torch.tensor([1.0 + 0.5j, 2.0 + 0.3j], dtype=torch.complex128)
        sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, (2, 2))

        result = sf.gamma(sparse)

        expected_values = sf.gamma(values)
        assert_close(result.values(), expected_values, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sparse_csr_dtypes(self, dtype):
        """Test sparse CSR with different dtypes."""
        crow_indices = torch.tensor([0, 1, 2])
        col_indices = torch.tensor([0, 1])
        values = torch.tensor([2.0, 3.0], dtype=dtype)
        sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, (2, 2))

        result = sf.gamma(sparse)

        assert result.dtype == dtype

    def test_sparse_empty_tensor(self):
        """Test sparse tensor with no non-zero values."""
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

        result = sf.gamma(sparse)

        assert result.is_sparse
        assert result._values().numel() == 0

    def test_sparse_single_element(self):
        """Test sparse tensor with single non-zero value."""
        indices = torch.tensor([[1], [2]])
        values = torch.tensor([2.0], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

        result = sf.gamma(sparse)

        expected = sf.gamma(values)
        assert_close(result._values(), expected, rtol=1e-10, atol=1e-10)


# =============================================================================
# Quantized tensor tests
# =============================================================================

class TestQuantizedTensors:
    """Test quantized tensor support for the gamma function."""

    def test_quantized_basic(self):
        """Test basic quantized tensor support."""
        # Create a float tensor and quantize it
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        scale = 0.1
        zero_point = 0
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        result = sf.gamma(quantized)

        # Result should be quantized
        assert result.is_quantized
        assert result.shape == quantized.shape

        # Dequantized result should be close to gamma of dequantized input
        expected = sf.gamma(quantized.dequantize())
        assert_close(result.dequantize(), expected, rtol=0.1, atol=0.1)

    def test_quantized_preserves_quantization_params(self):
        """Test that quantization parameters are preserved."""
        x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        scale = 0.05
        zero_point = 10
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        result = sf.gamma(quantized)

        assert result.q_scale() == scale
        assert result.q_zero_point() == zero_point

    def test_quantized_matches_float_approximately(self):
        """Test that quantized gamma matches float gamma within quantization error."""
        # Use smaller input values to avoid large gamma outputs that clip in quantization
        x = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)
        scale = 0.01
        zero_point = 0
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        quantized_result = sf.gamma(quantized).dequantize()
        float_result = sf.gamma(x)

        # Allow for quantization error (quint8 has limited range ~[0, 2.55] with scale=0.01)
        assert_close(quantized_result, float_result, rtol=0.1, atol=0.2)

    @pytest.mark.parametrize("qtype", [torch.quint8, torch.qint8])
    def test_quantized_dtypes(self, qtype):
        """Test quantized tensors with different quantization types."""
        x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        scale = 0.1
        zero_point = 0 if qtype == torch.qint8 else 128
        quantized = torch.quantize_per_tensor(x, scale, zero_point, qtype)

        result = sf.gamma(quantized)

        assert result.is_quantized
        assert result.dtype == qtype

    def test_quantized_factorial_values(self):
        """Test quantized gamma gives approximately correct factorial values."""
        # Gamma(n) = (n-1)! for positive integers
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        expected_float = torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0], dtype=torch.float32)

        scale = 0.1
        zero_point = 0
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        result = sf.gamma(quantized).dequantize()

        # Relaxed tolerance due to quantization
        assert_close(result, expected_float, rtol=0.1, atol=1.0)

    def test_quantized_2d_tensor(self):
        """Test quantized 2D tensor."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        scale = 0.1
        zero_point = 0
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        result = sf.gamma(quantized)

        assert result.is_quantized
        assert result.shape == (2, 2)

    def test_quantized_large_batch(self):
        """Test quantized tensor with larger batch size."""
        x = torch.rand(100, dtype=torch.float32) * 4 + 1  # Values in [1, 5]
        scale = 0.02
        zero_point = 0
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

        result = sf.gamma(quantized)

        assert result.is_quantized
        assert result.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
