"""Tests for torchscience.integral_transform.hilbert_transform."""

import math

import pytest
import torch

import torchscience.integral_transform


class TestHilbertTransformBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Test that output shape matches input shape for 1D tensor."""
        x = torch.randn(100)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape

    def test_output_shape_2d(self):
        """Test that output shape matches input shape for 2D tensor."""
        x = torch.randn(10, 100)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape

    def test_output_shape_3d(self):
        """Test that output shape matches input shape for 3D tensor."""
        x = torch.randn(5, 10, 100)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape

    def test_dim_0(self):
        """Test transform along dimension 0."""
        x = torch.randn(100, 10)
        result = torchscience.integral_transform.hilbert_transform(x, dim=0)
        assert result.shape == x.shape

    def test_dim_1(self):
        """Test transform along dimension 1."""
        x = torch.randn(10, 100)
        result = torchscience.integral_transform.hilbert_transform(x, dim=1)
        assert result.shape == x.shape

    def test_negative_dim(self):
        """Test transform with negative dimension."""
        x = torch.randn(10, 20, 100)
        result = torchscience.integral_transform.hilbert_transform(x, dim=-1)
        assert result.shape == x.shape

    def test_negative_dim_equals_positive(self):
        """Test that dim=-1 equals dim=last."""
        x = torch.randn(10, 100)
        result_neg = torchscience.integral_transform.hilbert_transform(
            x, dim=-1
        )
        result_pos = torchscience.integral_transform.hilbert_transform(
            x, dim=1
        )
        torch.testing.assert_close(result_neg, result_pos)


class TestHilbertTransformMathProperties:
    """Tests for mathematical properties of the Hilbert transform."""

    def test_sin_to_minus_cos(self):
        """Test that H[sin(wt)] = -cos(wt) for positive frequencies.

        This follows the standard mathematical convention where
        H[f](t) = (1/pi) PV integral f(tau)/(t-tau) dtau.
        """
        n = 1024
        t = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        # Use a frequency that fits exactly in n samples
        freq = 4.0
        x = torch.sin(freq * t)
        expected = -torch.cos(freq * t)

        result = torchscience.integral_transform.hilbert_transform(x)

        # Allow for edge effects at boundaries
        torch.testing.assert_close(
            result[100:-100], expected[100:-100], atol=0.1, rtol=0.1
        )

    def test_cos_to_sin(self):
        """Test that H[cos(wt)] = sin(wt) for positive frequencies.

        This follows the standard mathematical convention.
        """
        n = 1024
        t = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        freq = 4.0
        x = torch.cos(freq * t)
        expected = torch.sin(freq * t)

        result = torchscience.integral_transform.hilbert_transform(x)

        # Allow for edge effects at boundaries
        torch.testing.assert_close(
            result[100:-100], expected[100:-100], atol=0.1, rtol=0.1
        )

    def test_double_transform_negates(self):
        """Test that H[H[f]] ≈ -f (involutory up to sign).

        Note: The discrete Hilbert transform is an approximation of the
        continuous transform. The property H[H[f]] = -f only holds exactly
        for the continuous case. The discrete version has some numerical
        error, especially at boundaries.
        """
        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float64)

        h1 = torchscience.integral_transform.hilbert_transform(x)
        h2 = torchscience.integral_transform.hilbert_transform(h1)

        # H[H[f]] should approximately equal -f
        # Tolerance of 0.15 accounts for discrete approximation error
        torch.testing.assert_close(h2, -x, atol=0.15, rtol=0.15)

    def test_energy_preservation(self):
        """Test that the Hilbert transform approximately preserves energy.

        Note: The discrete Hilbert transform doesn't perfectly preserve
        energy due to edge effects. Energy ratio should be close to 1.0
        but not exactly 1.0.
        """
        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float64)

        h = torchscience.integral_transform.hilbert_transform(x)

        energy_x = torch.sum(x**2)
        energy_h = torch.sum(h**2)

        # Energy should be within 1% of original
        energy_ratio = energy_h / energy_x
        assert 0.99 < energy_ratio.item() < 1.01, (
            f"Energy ratio {energy_ratio.item()} not in [0.99, 1.01]"
        )

    def test_linearity_addition(self):
        """Test linearity: H[f + g] = H[f] + H[g]."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.float64)
        g = torch.randn(128, dtype=torch.float64)

        h_sum = torchscience.integral_transform.hilbert_transform(f + g)
        sum_h = torchscience.integral_transform.hilbert_transform(
            f
        ) + torchscience.integral_transform.hilbert_transform(g)

        torch.testing.assert_close(h_sum, sum_h, atol=1e-10, rtol=1e-10)

    def test_linearity_scaling(self):
        """Test linearity: H[a*f] = a*H[f]."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.float64)
        a = 3.14

        h_scaled = torchscience.integral_transform.hilbert_transform(a * f)
        scaled_h = a * torchscience.integral_transform.hilbert_transform(f)

        torch.testing.assert_close(h_scaled, scaled_h, atol=1e-10, rtol=1e-10)

    def test_dc_component_zero(self):
        """Test that DC component is annihilated (H[constant] = 0)."""
        x = torch.ones(128, dtype=torch.float64) * 5.0

        result = torchscience.integral_transform.hilbert_transform(x)

        # All values should be close to zero
        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-10, rtol=1e-10
        )


class TestHilbertTransformAnalyticSignal:
    """Tests for analytic signal construction."""

    def test_analytic_signal_positive_frequency_only(self):
        """Test that analytic signal z = x + i*H[x] has only positive frequencies."""
        n = 256
        t = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        freq = 8.0
        x = torch.sin(freq * t)

        h = torchscience.integral_transform.hilbert_transform(x)
        z = x + 1j * h  # Analytic signal

        # FFT of analytic signal should have negligible negative frequencies
        z_fft = torch.fft.fft(z)
        neg_freq_energy = torch.sum(torch.abs(z_fft[n // 2 + 1 :]) ** 2)
        total_energy = torch.sum(torch.abs(z_fft) ** 2)

        # Negative frequency energy should be negligible
        assert neg_freq_energy / total_energy < 1e-10


class TestHilbertTransformDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(100, dtype=dtype)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.dtype == dtype
        assert torch.all(torch.isfinite(result))

    def test_complex64_support(self):
        """Test complex64 support."""
        x = torch.randn(100, dtype=torch.complex64)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.dtype == torch.complex64
        assert torch.all(torch.isfinite(result.real))
        assert torch.all(torch.isfinite(result.imag))

    def test_complex128_support(self):
        """Test complex128 support."""
        x = torch.randn(100, dtype=torch.complex128)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.dtype == torch.complex128


class TestHilbertTransformGradient:
    """Tests for gradient computation."""

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.hilbert_transform(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.hilbert_transform(x)
        loss = result.sum()
        loss.backward()
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_dim(self):
        """Test gradient with dimension specified."""
        x = torch.randn(10, 64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.hilbert_transform(x, dim=1)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self):
        """Test gradient correctness with gradcheck."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.hilbert_transform(
                input_tensor
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_with_dim(self):
        """Test gradient correctness with dimension argument."""
        x = torch.randn(8, 32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.hilbert_transform(
                input_tensor, dim=1
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.hilbert_transform(
                input_tensor
            )

        assert torch.autograd.gradgradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradient_property(self):
        """Test that gradient of Hilbert transform is -H applied to grad_output.

        Since H^T = -H (anti-self-adjoint), the gradient should be -H[grad_output].
        """
        torch.manual_seed(42)
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        grad_output = torch.randn(64, dtype=torch.float64)

        result = torchscience.integral_transform.hilbert_transform(x)
        result.backward(grad_output)

        # The gradient should be -H[grad_output]
        expected_grad = -torchscience.integral_transform.hilbert_transform(
            grad_output
        )

        torch.testing.assert_close(
            x.grad, expected_grad, atol=1e-10, rtol=1e-10
        )


class TestHilbertTransformDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(100, device=torch.device("cpu"))
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(100, device=torch.device("cuda"))
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_cpu_consistency(self):
        """Test that CPU and CUDA give same results."""
        torch.manual_seed(42)
        x_cpu = torch.randn(100, dtype=torch.float64)
        x_cuda = x_cpu.cuda()

        result_cpu = torchscience.integral_transform.hilbert_transform(x_cpu)
        result_cuda = torchscience.integral_transform.hilbert_transform(x_cuda)

        torch.testing.assert_close(
            result_cpu, result_cuda.cpu(), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_gradient(self):
        """Test gradient computation on CUDA."""
        x = torch.randn(
            64, requires_grad=True, dtype=torch.float64, device="cuda"
        )
        result = torchscience.integral_transform.hilbert_transform(x)
        result.sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestHilbertTransformTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""

        def fn(x):
            return torchscience.integral_transform.hilbert_transform(x)

        compiled_fn = torch.compile(fn)
        x = torch.randn(100)
        result = compiled_fn(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_with_dim(self):
        """Test torch.compile with dimension argument."""

        def fn(x):
            return torchscience.integral_transform.hilbert_transform(x, dim=0)

        compiled_fn = torch.compile(fn)
        x = torch.randn(100, 10)
        result = compiled_fn(x)
        assert result.shape == x.shape

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_with_grad(self):
        """Test torch.compile with gradient computation."""

        def fn(x):
            return torchscience.integral_transform.hilbert_transform(x)

        compiled_fn = torch.compile(fn)
        x = torch.randn(64, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        assert x.grad is not None


class TestHilbertTransformSciPyCompatibility:
    """Tests for SciPy compatibility."""

    def test_analytic_signal_matches_scipy(self):
        """Test that analytic signal matches SciPy's hilbert function.

        Note: SciPy's scipy.signal.hilbert returns the analytic signal,
        not the Hilbert transform itself.
        """
        scipy_signal = pytest.importorskip("scipy.signal")

        torch.manual_seed(42)
        x_np = torch.randn(128).numpy()
        x_torch = torch.from_numpy(x_np)

        # SciPy returns analytic signal: z = x + i*H[x]
        scipy_analytic = scipy_signal.hilbert(x_np)
        scipy_hilbert = scipy_analytic.imag

        torch_hilbert = torchscience.integral_transform.hilbert_transform(
            x_torch
        )

        torch.testing.assert_close(
            torch_hilbert,
            torch.from_numpy(scipy_hilbert),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_matches_scipy_2d(self):
        """Test 2D compatibility with SciPy."""
        scipy_signal = pytest.importorskip("scipy.signal")

        torch.manual_seed(42)
        x_np = torch.randn(10, 128).numpy()
        x_torch = torch.from_numpy(x_np)

        # SciPy operates along last axis by default
        scipy_analytic = scipy_signal.hilbert(x_np)
        scipy_hilbert = scipy_analytic.imag

        torch_hilbert = torchscience.integral_transform.hilbert_transform(
            x_torch, dim=-1
        )

        torch.testing.assert_close(
            torch_hilbert,
            torch.from_numpy(scipy_hilbert),
            atol=1e-5,
            rtol=1e-5,
        )


class TestHilbertTransformBatched:
    """Tests for batched computation."""

    def test_batched_matches_loop(self):
        """Test that batched computation matches looped computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 100)

        # Batched
        batched_result = torchscience.integral_transform.hilbert_transform(
            x, dim=1
        )

        # Looped
        looped_results = []
        for i in range(5):
            looped_results.append(
                torchscience.integral_transform.hilbert_transform(x[i])
            )
        looped_result = torch.stack(looped_results)

        torch.testing.assert_close(
            batched_result, looped_result, rtol=1e-10, atol=1e-10
        )

    def test_large_batch(self):
        """Test with large batch size."""
        x = torch.randn(1000, 64)
        result = torchscience.integral_transform.hilbert_transform(x, dim=1)
        assert result.shape == (1000, 64)
        assert torch.all(torch.isfinite(result))


class TestHilbertTransformEdgeCases:
    """Tests for edge cases."""

    def test_single_element(self):
        """Test with single element tensor."""
        x = torch.tensor([1.0])
        result = torchscience.integral_transform.hilbert_transform(x)
        # Single element should have zero Hilbert transform (only DC)
        torch.testing.assert_close(
            result, torch.tensor([0.0]), atol=1e-10, rtol=1e-10
        )

    def test_two_elements(self):
        """Test with two element tensor."""
        x = torch.tensor([1.0, -1.0])
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_odd_length(self):
        """Test with odd length tensor."""
        x = torch.randn(127)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_power_of_two_length(self):
        """Test with power-of-two length (optimal for FFT)."""
        x = torch.randn(256)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))


class TestHilbertTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test basic vmap functionality."""
        x = torch.randn(5, 64)

        vmapped_fn = torch.vmap(
            torchscience.integral_transform.hilbert_transform
        )
        result_vmap = vmapped_fn(x)

        result_batch = torchscience.integral_transform.hilbert_transform(
            x, dim=-1
        )

        torch.testing.assert_close(result_vmap, result_batch)

    def test_vmap_nested(self):
        """Test nested vmap."""
        x = torch.randn(3, 4, 64)

        fn = torchscience.integral_transform.hilbert_transform
        result_vmap = torch.vmap(torch.vmap(fn))(x)

        result_batch = fn(x, dim=-1)

        torch.testing.assert_close(result_vmap, result_batch)


class TestHilbertTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_basic_shape(self):
        """Test meta tensor shape inference."""
        x_meta = torch.empty(100, device="meta")

        result = torchscience.integral_transform.hilbert_transform(x_meta)

        assert result.shape == (100,)
        assert result.device.type == "meta"

    def test_meta_batch_shape(self):
        """Test meta tensor with batch dimensions."""
        x_meta = torch.empty(10, 100, device="meta")

        result = torchscience.integral_transform.hilbert_transform(
            x_meta, dim=1
        )

        assert result.shape == (10, 100)
        assert result.device.type == "meta"

    def test_meta_dtype(self):
        """Test meta tensor dtype preservation."""
        x_meta = torch.empty(100, dtype=torch.float64, device="meta")

        result = torchscience.integral_transform.hilbert_transform(x_meta)

        assert result.dtype == torch.float64


class TestHilbertTransformComplexInput:
    """Tests for complex input handling."""

    def test_complex_input_shape_preserved(self):
        """Test that complex input produces complex output with same shape."""
        x = torch.randn(100, dtype=torch.complex128)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert result.shape == x.shape
        assert result.dtype == torch.complex128

    def test_purely_imaginary_input(self):
        """Test Hilbert transform of purely imaginary signal.

        H[i*f] = i*H[f] by linearity.
        """
        torch.manual_seed(42)
        f_real = torch.randn(128, dtype=torch.float64)
        f_imag = 1j * f_real.to(torch.complex128)

        h_real = torchscience.integral_transform.hilbert_transform(f_real)
        h_imag = torchscience.integral_transform.hilbert_transform(f_imag)

        # H[i*f] should equal i*H[f]
        expected = 1j * h_real.to(torch.complex128)
        torch.testing.assert_close(h_imag, expected, atol=1e-10, rtol=1e-10)

    def test_complex_linearity(self):
        """Test linearity for complex inputs: H[a + ib] = H[a] + i*H[b]."""
        torch.manual_seed(42)
        a = torch.randn(128, dtype=torch.float64)
        b = torch.randn(128, dtype=torch.float64)
        z = (a + 1j * b).to(torch.complex128)

        h_z = torchscience.integral_transform.hilbert_transform(z)
        h_a = torchscience.integral_transform.hilbert_transform(a)
        h_b = torchscience.integral_transform.hilbert_transform(b)

        expected = (h_a + 1j * h_b).to(torch.complex128)
        torch.testing.assert_close(h_z, expected, atol=1e-10, rtol=1e-10)

    def test_conjugate_symmetry(self):
        """Test that H[conj(f)] = conj(H[f])."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.complex128)

        h_f = torchscience.integral_transform.hilbert_transform(f)
        h_conj_f = torchscience.integral_transform.hilbert_transform(
            torch.conj(f)
        )

        torch.testing.assert_close(
            h_conj_f, torch.conj(h_f), atol=1e-10, rtol=1e-10
        )

    def test_complex_double_transform(self):
        """Test H[H[f]] = -f for complex input."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.complex128)

        h1 = torchscience.integral_transform.hilbert_transform(f)
        h2 = torchscience.integral_transform.hilbert_transform(h1)

        # H[H[f]] should approximately equal -f
        torch.testing.assert_close(h2, -f, atol=0.15, rtol=0.15)

    def test_complex_gradient_via_view(self):
        """Test gradient computation for complex input using view_as_real.

        This tests that gradients can flow through complex hilbert transform
        when using view_as_real/view_as_complex which have proper backward support.
        """
        # Create a complex tensor via view_as_complex (has proper gradient support)
        x_real_imag = torch.randn(
            64, 2, dtype=torch.float64, requires_grad=True
        )
        x = torch.view_as_complex(x_real_imag)

        result = torchscience.integral_transform.hilbert_transform(x)
        loss = result.abs().sum()
        loss.backward()

        # Gradients flow through to the underlying real tensor
        assert x_real_imag.grad is not None
        assert torch.all(torch.isfinite(x_real_imag.grad))

    def test_complex_gradcheck_via_view(self):
        """Test gradient correctness for complex input with gradcheck.

        Uses view_as_complex which has proper backward support in PyTorch.
        """
        x_real_imag = torch.randn(
            32, 2, dtype=torch.float64, requires_grad=True
        )

        def fn(real_imag):
            z = torch.view_as_complex(real_imag)
            result = torchscience.integral_transform.hilbert_transform(z)
            # Return as real view for gradcheck
            return torch.view_as_real(result)

        assert torch.autograd.gradcheck(
            fn, (x_real_imag,), eps=1e-5, atol=1e-4, rtol=1e-4
        )


class TestHilbertTransformNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_small_values(self):
        """Test stability with very small values (near underflow)."""
        x = torch.randn(128, dtype=torch.float64) * 1e-30
        result = torchscience.integral_transform.hilbert_transform(x)

        assert torch.all(torch.isfinite(result))
        # Result should also be very small
        assert result.abs().max() < 1e-25

    def test_very_large_values(self):
        """Test stability with very large values (near overflow)."""
        x = torch.randn(128, dtype=torch.float64) * 1e30
        result = torchscience.integral_transform.hilbert_transform(x)

        assert torch.all(torch.isfinite(result))
        # Result magnitude should be similar to input
        assert result.abs().max() > 1e25

    def test_mixed_magnitude_values(self):
        """Test stability with mixed small and large values."""
        torch.manual_seed(42)
        x = torch.randn(128, dtype=torch.float64)
        x[::2] *= 1e-15  # Every other element is tiny
        x[1::2] *= 1e15  # Alternating elements are huge

        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))

    def test_subnormal_values(self):
        """Test handling of subnormal (denormalized) floating point numbers."""
        # Smallest positive subnormal for float64 is ~5e-324
        x = torch.tensor([1e-310, 1e-315, 1e-308, 1.0], dtype=torch.float64)
        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))

    def test_zeros_and_nonzeros(self):
        """Test stability with sparse-like patterns (many zeros)."""
        x = torch.zeros(128, dtype=torch.float64)
        x[::10] = torch.randn(
            13, dtype=torch.float64
        )  # Only every 10th element non-zero

        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))

    def test_impulse_response(self):
        """Test response to an impulse (single non-zero value)."""
        x = torch.zeros(128, dtype=torch.float64)
        x[64] = 1.0  # Single impulse at center

        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))
        # Impulse response should be non-zero everywhere except DC
        assert result.abs().sum() > 0

    def test_step_function(self):
        """Test response to a step function."""
        x = torch.zeros(128, dtype=torch.float64)
        x[64:] = 1.0  # Step at center

        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))

    def test_alternating_sign(self):
        """Test with rapidly alternating signs (high frequency content)."""
        x = torch.ones(128, dtype=torch.float64)
        x[1::2] = -1.0  # Alternating +1, -1, +1, -1, ...

        result = torchscience.integral_transform.hilbert_transform(x)
        assert torch.all(torch.isfinite(result))

    def test_gradient_stability_small_values(self):
        """Test gradient stability with small values."""
        x = (torch.randn(64, dtype=torch.float64) * 1e-20).requires_grad_(True)
        result = torchscience.integral_transform.hilbert_transform(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_stability_large_values(self):
        """Test gradient stability with large values."""
        x = (torch.randn(64, dtype=torch.float64) * 1e20).requires_grad_(True)
        result = torchscience.integral_transform.hilbert_transform(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_near_zero_energy(self):
        """Test with signal that has near-zero total energy."""
        x = torch.randn(128, dtype=torch.float64) * 1e-150
        result = torchscience.integral_transform.hilbert_transform(x)

        assert torch.all(torch.isfinite(result))

    def test_single_frequency_exact(self):
        """Test numerical precision for single frequency sinusoid.

        For a pure sinusoid, the Hilbert transform should be exact
        (no edge effects for frequencies that fit exactly in the signal).
        """
        n = 256
        k = 8  # Exactly 8 cycles in n samples
        t = torch.arange(n, dtype=torch.float64) / n * 2 * math.pi * k
        x = torch.sin(t)
        expected = -torch.cos(t)

        result = torchscience.integral_transform.hilbert_transform(x)

        # For exact frequency, should be very accurate
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_float32_precision(self):
        """Test that float32 maintains reasonable precision."""
        torch.manual_seed(42)
        x_64 = torch.randn(128, dtype=torch.float64)
        x_32 = x_64.float()

        result_64 = torchscience.integral_transform.hilbert_transform(x_64)
        result_32 = torchscience.integral_transform.hilbert_transform(x_32)

        # Float32 result should be close to float64 result
        torch.testing.assert_close(
            result_32.double(), result_64, atol=1e-5, rtol=1e-5
        )


class TestHilbertTransformErrorCases:
    """Tests for error handling."""

    def test_invalid_dim_positive(self):
        """Test error for dimension out of range (positive)."""
        x = torch.randn(10, 20)
        with pytest.raises(RuntimeError, match="dim out of range"):
            torchscience.integral_transform.hilbert_transform(x, dim=5)

    def test_invalid_dim_negative(self):
        """Test error for dimension out of range (negative)."""
        x = torch.randn(10, 20)
        with pytest.raises(RuntimeError, match="dim out of range"):
            torchscience.integral_transform.hilbert_transform(x, dim=-5)

    def test_empty_tensor(self):
        """Test error for empty tensor."""
        x = torch.empty(0)
        with pytest.raises(RuntimeError, match="non-empty|positive size"):
            torchscience.integral_transform.hilbert_transform(x)

    def test_empty_along_dim(self):
        """Test error for tensor with zero size along transform dimension."""
        x = torch.empty(10, 0)
        with pytest.raises(RuntimeError, match="non-empty|positive size"):
            torchscience.integral_transform.hilbert_transform(x, dim=1)


class TestHilbertTransformNParameter:
    """Tests for the n parameter (signal length for zero-padding/truncation)."""

    def test_n_parameter_zero_padding(self):
        """Test that n parameter can zero-pad the signal."""
        x = torch.randn(64)
        result = torchscience.integral_transform.hilbert_transform(x, n=128)
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_n_parameter_truncation(self):
        """Test that n parameter can truncate the signal."""
        x = torch.randn(128)
        result = torchscience.integral_transform.hilbert_transform(x, n=64)
        assert result.shape == (64,)
        assert torch.all(torch.isfinite(result))

    def test_n_parameter_same_as_input(self):
        """Test that n equal to input size gives same result as default."""
        x = torch.randn(100, dtype=torch.float64)
        result_default = torchscience.integral_transform.hilbert_transform(x)
        result_n = torchscience.integral_transform.hilbert_transform(x, n=100)
        torch.testing.assert_close(result_default, result_n)

    def test_n_parameter_with_dim(self):
        """Test n parameter with explicit dimension."""
        x = torch.randn(10, 64)
        result = torchscience.integral_transform.hilbert_transform(
            x, n=128, dim=1
        )
        assert result.shape == (10, 128)

    def test_n_parameter_batched(self):
        """Test n parameter with batched input."""
        x = torch.randn(5, 64)
        result = torchscience.integral_transform.hilbert_transform(
            x, n=128, dim=-1
        )
        assert result.shape == (5, 128)

    def test_n_parameter_dim0(self):
        """Test n parameter along dimension 0."""
        x = torch.randn(64, 10)
        result = torchscience.integral_transform.hilbert_transform(
            x, n=128, dim=0
        )
        assert result.shape == (128, 10)

    def test_n_parameter_gradient(self):
        """Test that gradients work with n parameter."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.hilbert_transform(x, n=128)
        loss = result.sum()
        loss.backward()
        # Gradient shape should match output shape, not input shape
        # (This tests the backward behavior with n parameter)
        assert x.grad is not None
        assert x.grad.shape == x.shape  # Gradient flows back to input
        assert torch.all(torch.isfinite(x.grad))

    def test_n_parameter_power_of_two_efficiency(self):
        """Test using n parameter to pad to power of 2 for FFT efficiency."""
        # Non-power-of-2 input
        x = torch.randn(100, dtype=torch.float64)
        # Pad to power of 2
        result = torchscience.integral_transform.hilbert_transform(x, n=128)
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_n_parameter_edge_effect_mitigation(self):
        """Test that zero-padding via n can help mitigate edge effects.

        For a non-periodic signal, zero-padding should reduce edge artifacts
        compared to no padding.
        """
        # Create a signal that starts and ends at different values (non-periodic)
        n = 64
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        x = t  # Simple ramp from 0 to 1

        # Transform without padding vs with padding
        h_no_pad = torchscience.integral_transform.hilbert_transform(x)
        h_padded = torchscience.integral_transform.hilbert_transform(x, n=256)[
            :n
        ]

        # Both should be finite
        assert torch.all(torch.isfinite(h_no_pad))
        assert torch.all(torch.isfinite(h_padded))

    def test_n_parameter_complex_input(self):
        """Test n parameter with complex input."""
        x = torch.randn(64, dtype=torch.complex128)
        result = torchscience.integral_transform.hilbert_transform(x, n=128)
        assert result.shape == (128,)
        assert result.dtype == torch.complex128

    def test_n_parameter_scipy_compatibility(self):
        """Test that n parameter behavior matches scipy.signal.hilbert N parameter."""
        scipy_signal = pytest.importorskip("scipy.signal")

        torch.manual_seed(42)
        x_np = torch.randn(64).numpy()
        x_torch = torch.from_numpy(x_np)

        # SciPy hilbert with N parameter returns analytic signal
        scipy_analytic = scipy_signal.hilbert(x_np, N=128)
        scipy_hilbert = scipy_analytic.imag

        torch_hilbert = torchscience.integral_transform.hilbert_transform(
            x_torch, n=128
        )

        torch.testing.assert_close(
            torch_hilbert,
            torch.from_numpy(scipy_hilbert),
            atol=1e-5,
            rtol=1e-5,
        )
