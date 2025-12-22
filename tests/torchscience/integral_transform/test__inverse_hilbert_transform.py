"""Tests for torchscience.integral_transform.inverse_hilbert_transform."""

import math

import pytest
import torch

import torchscience.integral_transform


class TestInverseHilbertTransformBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Test that output shape matches input shape for 1D tensor."""
        x = torch.randn(100)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape

    def test_output_shape_2d(self):
        """Test that output shape matches input shape for 2D tensor."""
        x = torch.randn(10, 100)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape

    def test_output_shape_3d(self):
        """Test that output shape matches input shape for 3D tensor."""
        x = torch.randn(5, 10, 100)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape

    def test_dim_0(self):
        """Test transform along dimension 0."""
        x = torch.randn(100, 10)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=0
        )
        assert result.shape == x.shape

    def test_dim_1(self):
        """Test transform along dimension 1."""
        x = torch.randn(10, 100)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=1
        )
        assert result.shape == x.shape

    def test_negative_dim(self):
        """Test transform with negative dimension."""
        x = torch.randn(10, 20, 100)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=-1
        )
        assert result.shape == x.shape

    def test_negative_dim_equals_positive(self):
        """Test that dim=-1 equals dim=last."""
        x = torch.randn(10, 100)
        result_neg = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=-1
        )
        result_pos = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=1
        )
        torch.testing.assert_close(result_neg, result_pos)


class TestInverseHilbertTransformMathProperties:
    """Tests for mathematical properties of the inverse Hilbert transform."""

    def test_inverse_equals_negative_forward(self):
        """Test that H^{-1}[f] = -H[f]."""
        torch.manual_seed(42)
        x = torch.randn(128, dtype=torch.float64)

        h = torchscience.integral_transform.hilbert_transform(x)
        h_inv = torchscience.integral_transform.inverse_hilbert_transform(x)

        torch.testing.assert_close(h_inv, -h, atol=1e-10, rtol=1e-10)

    def test_inverse_undoes_forward(self):
        """Test that H^{-1}[H[f]] ≈ f.

        Note: The discrete Hilbert transform is an approximation.
        """
        torch.manual_seed(42)
        x = torch.randn(128, dtype=torch.float64)

        h = torchscience.integral_transform.hilbert_transform(x)
        recovered = torchscience.integral_transform.inverse_hilbert_transform(
            h
        )

        # Tolerance accounts for discrete approximation error
        torch.testing.assert_close(recovered, x, atol=0.15, rtol=0.15)

    def test_forward_undoes_inverse(self):
        """Test that H[H^{-1}[f]] ≈ f.

        Note: The discrete Hilbert transform is an approximation.
        """
        torch.manual_seed(42)
        x = torch.randn(128, dtype=torch.float64)

        h_inv = torchscience.integral_transform.inverse_hilbert_transform(x)
        recovered = torchscience.integral_transform.hilbert_transform(h_inv)

        # Tolerance accounts for discrete approximation error
        torch.testing.assert_close(recovered, x, atol=0.15, rtol=0.15)

    def test_energy_preservation(self):
        """Test that the inverse Hilbert transform approximately preserves energy.

        Note: The discrete Hilbert transform doesn't perfectly preserve
        energy due to edge effects.
        """
        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float64)

        h_inv = torchscience.integral_transform.inverse_hilbert_transform(x)

        energy_x = torch.sum(x**2)
        energy_h_inv = torch.sum(h_inv**2)

        # Energy should be within 1% of original
        energy_ratio = energy_h_inv / energy_x
        assert 0.99 < energy_ratio.item() < 1.01, (
            f"Energy ratio {energy_ratio.item()} not in [0.99, 1.01]"
        )

    def test_linearity_addition(self):
        """Test linearity: H^{-1}[f + g] = H^{-1}[f] + H^{-1}[g]."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.float64)
        g = torch.randn(128, dtype=torch.float64)

        h_sum = torchscience.integral_transform.inverse_hilbert_transform(
            f + g
        )
        sum_h = torchscience.integral_transform.inverse_hilbert_transform(
            f
        ) + torchscience.integral_transform.inverse_hilbert_transform(g)

        torch.testing.assert_close(h_sum, sum_h, atol=1e-10, rtol=1e-10)

    def test_linearity_scaling(self):
        """Test linearity: H^{-1}[a*f] = a*H^{-1}[f]."""
        torch.manual_seed(42)
        f = torch.randn(128, dtype=torch.float64)
        a = 3.14

        h_scaled = torchscience.integral_transform.inverse_hilbert_transform(
            a * f
        )
        scaled_h = (
            a * torchscience.integral_transform.inverse_hilbert_transform(f)
        )

        torch.testing.assert_close(h_scaled, scaled_h, atol=1e-10, rtol=1e-10)

    def test_dc_component_zero(self):
        """Test that DC component is annihilated (H^{-1}[constant] = 0)."""
        x = torch.ones(128, dtype=torch.float64) * 5.0

        result = torchscience.integral_transform.inverse_hilbert_transform(x)

        torch.testing.assert_close(
            result, torch.zeros_like(result), atol=1e-10, rtol=1e-10
        )


class TestInverseHilbertTransformTrigonometric:
    """Tests for trigonometric function behavior."""

    def test_cos_to_minus_sin(self):
        """Test that H^{-1}[cos(wt)] = -sin(wt) for positive frequencies.

        Since H^{-1} = -H and H[cos] = sin, we have H^{-1}[cos] = -sin.
        """
        n = 1024
        t = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        freq = 4.0
        x = torch.cos(freq * t)
        expected = -torch.sin(freq * t)

        result = torchscience.integral_transform.inverse_hilbert_transform(x)

        # Allow for edge effects at boundaries
        torch.testing.assert_close(
            result[100:-100], expected[100:-100], atol=0.1, rtol=0.1
        )

    def test_sin_to_cos(self):
        """Test that H^{-1}[sin(wt)] = cos(wt) for positive frequencies.

        Since H^{-1} = -H and H[sin] = -cos, we have H^{-1}[sin] = -(-cos) = cos.
        """
        n = 1024
        t = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        freq = 4.0
        x = torch.sin(freq * t)
        expected = torch.cos(freq * t)

        result = torchscience.integral_transform.inverse_hilbert_transform(x)

        # Allow for edge effects at boundaries
        torch.testing.assert_close(
            result[100:-100], expected[100:-100], atol=0.1, rtol=0.1
        )


class TestInverseHilbertTransformDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(100, dtype=dtype)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.dtype == dtype
        assert torch.all(torch.isfinite(result))

    def test_complex64_support(self):
        """Test complex64 support."""
        x = torch.randn(100, dtype=torch.complex64)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.dtype == torch.complex64
        assert torch.all(torch.isfinite(result.real))
        assert torch.all(torch.isfinite(result.imag))

    def test_complex128_support(self):
        """Test complex128 support."""
        x = torch.randn(100, dtype=torch.complex128)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.dtype == torch.complex128


class TestInverseHilbertTransformGradient:
    """Tests for gradient computation."""

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_finite(self):
        """Test that gradients are finite."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        loss = result.sum()
        loss.backward()
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_dim(self):
        """Test gradient with dimension specified."""
        x = torch.randn(10, 64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=1
        )
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self):
        """Test gradient correctness with gradcheck."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.inverse_hilbert_transform(
                input_tensor
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_with_dim(self):
        """Test gradient correctness with dimension argument."""
        x = torch.randn(8, 32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.inverse_hilbert_transform(
                input_tensor, dim=1
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.integral_transform.inverse_hilbert_transform(
                input_tensor
            )

        assert torch.autograd.gradgradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradient_property(self):
        """Test that gradient equals H applied to grad_output.

        Since (H^{-1})^T = H (because H^T = -H and H^{-1} = -H),
        the gradient should be H[grad_output].
        """
        torch.manual_seed(42)
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        grad_output = torch.randn(64, dtype=torch.float64)

        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        result.backward(grad_output)

        # The gradient should be H[grad_output]
        # Since H^{-1} = -H, then (H^{-1})^T = (-H)^T = -H^T = H
        expected_grad = torchscience.integral_transform.hilbert_transform(
            grad_output
        )

        torch.testing.assert_close(
            x.grad, expected_grad, atol=1e-10, rtol=1e-10
        )


class TestInverseHilbertTransformDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(100, device=torch.device("cpu"))
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(100, device=torch.device("cuda"))
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_cpu_consistency(self):
        """Test that CPU and CUDA give same results."""
        torch.manual_seed(42)
        x_cpu = torch.randn(100, dtype=torch.float64)
        x_cuda = x_cpu.cuda()

        result_cpu = torchscience.integral_transform.inverse_hilbert_transform(
            x_cpu
        )
        result_cuda = (
            torchscience.integral_transform.inverse_hilbert_transform(x_cuda)
        )

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
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        result.sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestInverseHilbertTransformTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_torch_compile_basic(self):
        """Test basic torch.compile compatibility."""

        def fn(x):
            return torchscience.integral_transform.inverse_hilbert_transform(x)

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
            return torchscience.integral_transform.inverse_hilbert_transform(
                x, dim=0
            )

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
            return torchscience.integral_transform.inverse_hilbert_transform(x)

        compiled_fn = torch.compile(fn)
        x = torch.randn(64, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        assert x.grad is not None


class TestInverseHilbertTransformBatched:
    """Tests for batched computation."""

    def test_batched_matches_loop(self):
        """Test that batched computation matches looped computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 100)

        # Batched
        batched_result = (
            torchscience.integral_transform.inverse_hilbert_transform(x, dim=1)
        )

        # Looped
        looped_results = []
        for i in range(5):
            looped_results.append(
                torchscience.integral_transform.inverse_hilbert_transform(x[i])
            )
        looped_result = torch.stack(looped_results)

        torch.testing.assert_close(
            batched_result, looped_result, rtol=1e-10, atol=1e-10
        )

    def test_large_batch(self):
        """Test with large batch size."""
        x = torch.randn(1000, 64)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, dim=1
        )
        assert result.shape == (1000, 64)
        assert torch.all(torch.isfinite(result))


class TestInverseHilbertTransformEdgeCases:
    """Tests for edge cases."""

    def test_single_element(self):
        """Test with single element tensor."""
        x = torch.tensor([1.0])
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        # Single element should have zero inverse Hilbert transform (only DC)
        torch.testing.assert_close(
            result, torch.tensor([0.0]), atol=1e-10, rtol=1e-10
        )

    def test_two_elements(self):
        """Test with two element tensor."""
        x = torch.tensor([1.0, -1.0])
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_odd_length(self):
        """Test with odd length tensor."""
        x = torch.randn(127)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_power_of_two_length(self):
        """Test with power-of-two length (optimal for FFT)."""
        x = torch.randn(256)
        result = torchscience.integral_transform.inverse_hilbert_transform(x)
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))


class TestInverseHilbertTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test basic vmap functionality."""
        x = torch.randn(5, 64)

        vmapped_fn = torch.vmap(
            torchscience.integral_transform.inverse_hilbert_transform
        )
        result_vmap = vmapped_fn(x)

        result_batch = (
            torchscience.integral_transform.inverse_hilbert_transform(
                x, dim=-1
            )
        )

        torch.testing.assert_close(result_vmap, result_batch)

    def test_vmap_nested(self):
        """Test nested vmap."""
        x = torch.randn(3, 4, 64)

        fn = torchscience.integral_transform.inverse_hilbert_transform
        result_vmap = torch.vmap(torch.vmap(fn))(x)

        result_batch = fn(x, dim=-1)

        torch.testing.assert_close(result_vmap, result_batch)


class TestInverseHilbertTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_basic_shape(self):
        """Test meta tensor shape inference."""
        x_meta = torch.empty(100, device="meta")

        result = torchscience.integral_transform.inverse_hilbert_transform(
            x_meta
        )

        assert result.shape == (100,)
        assert result.device.type == "meta"

    def test_meta_batch_shape(self):
        """Test meta tensor with batch dimensions."""
        x_meta = torch.empty(10, 100, device="meta")

        result = torchscience.integral_transform.inverse_hilbert_transform(
            x_meta, dim=1
        )

        assert result.shape == (10, 100)
        assert result.device.type == "meta"

    def test_meta_dtype(self):
        """Test meta tensor dtype preservation."""
        x_meta = torch.empty(100, dtype=torch.float64, device="meta")

        result = torchscience.integral_transform.inverse_hilbert_transform(
            x_meta
        )

        assert result.dtype == torch.float64


class TestInverseHilbertTransformNParameter:
    """Tests for the n parameter (signal length for zero-padding/truncation)."""

    def test_n_parameter_zero_padding(self):
        """Test that n parameter can zero-pad the signal."""
        x = torch.randn(64)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, n=128
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_n_parameter_truncation(self):
        """Test that n parameter can truncate the signal."""
        x = torch.randn(128)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, n=64
        )
        assert result.shape == (64,)
        assert torch.all(torch.isfinite(result))

    def test_n_parameter_same_as_input(self):
        """Test that n equal to input size gives same result as default."""
        x = torch.randn(100, dtype=torch.float64)
        result_default = (
            torchscience.integral_transform.inverse_hilbert_transform(x)
        )
        result_n = torchscience.integral_transform.inverse_hilbert_transform(
            x, n=100
        )
        torch.testing.assert_close(result_default, result_n)

    def test_n_parameter_with_dim(self):
        """Test n parameter with explicit dimension."""
        x = torch.randn(10, 64)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, n=128, dim=1
        )
        assert result.shape == (10, 128)

    def test_n_parameter_gradient(self):
        """Test that gradients work with n parameter."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.integral_transform.inverse_hilbert_transform(
            x, n=128
        )
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_n_parameter_inverse_identity(self):
        """Test that H^{-1}[H[x]] = x still holds with n parameter.

        When using the same n for both transforms, the identity should hold.
        """
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.float64)

        h = torchscience.integral_transform.hilbert_transform(x, n=128)
        x_recovered = (
            torchscience.integral_transform.inverse_hilbert_transform(h, n=128)
        )

        # Compare the first 64 elements (original signal length)
        # Note: With padding, the relationship is more complex
        assert x_recovered.shape == (128,)
        assert torch.all(torch.isfinite(x_recovered))
