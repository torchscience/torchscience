import pytest
import torch
import torchscience
from torchscience import ops
from torch.library import opcheck
from torch.autograd import gradcheck, gradgradcheck


class TestExampleOperator:
    """Test suite for the example operator (adds scalar to all elements)."""

    @pytest.fixture(params=["cpu"])
    def device(self, request):
        """Parametrize tests across available devices."""
        return request.param

    @pytest.fixture(params=[torch.float32, torch.float64, torch.int32, torch.int64])
    def dtype(self, request):
        """Parametrize tests across different data types."""
        return request.param

    def test_basic_addition(self, device, dtype):
        """Test that the operator performs scalar addition."""
        if dtype in [torch.int32, torch.int64]:
            x = torch.randint(0, 100, (10, 20), dtype=dtype, device=device)
        else:
            x = torch.randn(10, 20, dtype=dtype, device=device)

        scalar_val = 5
        result = ops.example(x, scalar_val)

        assert result.shape == x.shape
        assert result.dtype == x.dtype
        assert result.device == x.device

        expected = x + scalar_val
        if dtype.is_floating_point:
            assert torch.allclose(result, expected)
        else:
            assert torch.equal(result, expected)

    def test_empty_tensor(self, device):
        """Test with empty tensor."""
        x = torch.empty(0, dtype=torch.float32, device=device)
        result = ops.example(x, 1.0)

        assert result.shape == x.shape
        assert result.numel() == 0

    def test_scalar_tensor(self, device):
        """Test with scalar tensor (0-dimensional)."""
        x = torch.tensor(5.0, device=device)
        result = ops.example(x, 3.0)

        assert result.shape == x.shape
        assert torch.allclose(result, torch.tensor(8.0, device=device))

    def test_large_tensor(self, device):
        """Test with large tensor."""
        x = torch.randn(100, 100, 100, device=device)
        scalar_val = 2.5
        result = ops.example(x, scalar_val)

        assert result.shape == x.shape
        assert torch.allclose(result, x + scalar_val)

    def test_different_shapes(self, device):
        """Test with various tensor shapes."""
        shapes = [
            (1,),           # 1D
            (10,),          # 1D larger
            (5, 5),         # 2D square
            (3, 7),         # 2D rectangular
            (2, 3, 4),      # 3D
            (2, 3, 4, 5),   # 4D
            (1, 2, 3, 4, 5), # 5D
        ]

        scalar_val = 1.5
        for shape in shapes:
            x = torch.randn(*shape, device=device)
            result = ops.example(x, scalar_val)
            assert result.shape == x.shape
            assert torch.allclose(result, x + scalar_val)

    def test_different_scalar_values(self, device):
        """Test with different scalar values."""
        x = torch.randn(10, 10, device=device)

        # Test various scalar values
        scalar_values = [0.0, 1.0, -1.0, 2.5, -3.7, 100.0, -200.5]

        for scalar_val in scalar_values:
            result = ops.example(x, scalar_val)
            assert torch.allclose(result, x + scalar_val)

    def test_contiguous_and_non_contiguous(self, device):
        """Test with both contiguous and non-contiguous tensors."""
        x = torch.randn(10, 20, device=device)
        scalar_val = 3.0

        # Test contiguous
        assert x.is_contiguous()
        result_contiguous = ops.example(x, scalar_val)
        assert torch.allclose(result_contiguous, x + scalar_val)

        # Test non-contiguous (transposed)
        x_t = x.t()
        assert not x_t.is_contiguous()
        result_non_contiguous = ops.example(x_t, scalar_val)
        assert torch.allclose(result_non_contiguous, x_t + scalar_val)

    def test_autograd(self):
        """Test autograd functionality."""
        x = torch.randn(10, 10, requires_grad=True)
        scalar_val = 5.0
        result = ops.example(x, scalar_val)

        # Compute loss and backward
        loss = result.sum()
        loss.backward()

        # For addition, gradient with respect to input should be all ones
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_autograd_chain(self):
        """Test autograd with chain of operations."""
        x = torch.randn(5, 5, requires_grad=True)

        # Chain multiple operations
        y = ops.example(x, 2.0)
        z = y * 3
        w = ops.example(z, 1.0)
        loss = w.sum()

        loss.backward()

        # Gradient should be 3 * ones (due to multiplication by 3)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x) * 3)

    def test_autograd_multiple_scalars(self):
        """Test autograd with different scalar values."""
        for scalar_val in [0.0, 1.0, -1.0, 5.5]:
            x = torch.randn(5, 5, requires_grad=True)
            result = ops.example(x, scalar_val)
            loss = result.sum()
            loss.backward()

            # Gradient should always be ones regardless of scalar value
            assert x.grad is not None
            assert torch.allclose(x.grad, torch.ones_like(x))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test CUDA backend."""
        x = torch.randn(100, 100, device='cuda')
        scalar_val = 2.5
        result = ops.example(x, scalar_val)

        assert result.device.type == 'cuda'
        assert torch.allclose(result.cpu(), (x + scalar_val).cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_half_precision(self):
        """Test CUDA backend with half precision."""
        x = torch.randn(100, 100, dtype=torch.float16, device='cuda')
        scalar_val = 1.5
        result = ops.example(x, scalar_val)

        assert result.dtype == torch.float16
        assert torch.allclose(result, x + scalar_val)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps(self):
        """Test MPS backend (Apple Silicon)."""
        x = torch.randn(100, 100, device='mps')
        scalar_val = 3.0
        result = ops.example(x, scalar_val)

        assert result.device.type == 'mps'
        assert torch.allclose(result.cpu(), (x + scalar_val).cpu())

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_multiple_dtypes(self):
        """Test MPS backend with multiple data types."""
        dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

        for dtype in dtypes:
            scalar_val = 5
            if dtype in [torch.float32, torch.float16]:
                x = torch.randn(50, 50, dtype=dtype, device='mps')
            else:
                x = torch.randint(0, 100, (50, 50), dtype=dtype, device='mps')

            result = ops.example(x, scalar_val)

            assert result.dtype == dtype
            expected = x + scalar_val
            assert torch.equal(result, expected)

    def test_inplace_safety(self, device):
        """Test that the operator doesn't modify input in-place."""
        x = torch.randn(10, 10, device=device)
        x_original = x.clone()
        scalar_val = 7.0

        result = ops.example(x, scalar_val)

        # Input should remain unchanged
        assert torch.allclose(x, x_original)
        # Result should be input + scalar
        assert torch.allclose(result, x_original + scalar_val)

    def test_deterministic(self, device):
        """Test that the operator is deterministic."""
        x = torch.randn(10, 10, device=device)
        scalar_val = 4.0

        result1 = ops.example(x, scalar_val)
        result2 = ops.example(x, scalar_val)

        assert torch.allclose(result1, result2)

    def test_memory_format(self, device):
        """Test with different memory formats."""
        x = torch.randn(2, 3, 4, 5, device=device)
        scalar_val = 2.0

        # Test with channels last
        x_channels_last = x.contiguous(memory_format=torch.channels_last)
        result = ops.example(x_channels_last, scalar_val)

        assert torch.allclose(result, x_channels_last + scalar_val)

    def test_negative_scalar(self, device):
        """Test with negative scalar values."""
        x = torch.randn(10, 10, device=device)
        scalar_val = -5.0

        result = ops.example(x, scalar_val)
        assert torch.allclose(result, x + scalar_val)

    def test_zero_scalar(self, device):
        """Test with zero scalar (should be identity)."""
        x = torch.randn(10, 10, device=device)
        result = ops.example(x, 0.0)

        assert torch.allclose(result, x)


def test_quantized_cpu():
    """Test quantized CPU backend."""
    # Create a regular float tensor
    x_float = torch.randn(10, 10)

    # Quantize it to int8
    scale = 0.1
    zero_point = 0
    x_quantized = torch.quantize_per_tensor(x_float, scale, zero_point, torch.qint8)

    # Apply the operator
    scalar_val = 5.0
    result = ops.example(x_quantized, scalar_val)

    # Check that scale and zero_point are preserved
    assert result.q_scale() == x_quantized.q_scale()
    assert result.q_zero_point() == x_quantized.q_zero_point()
    assert result.dtype == x_quantized.dtype

    # Dequantize to check the actual values
    x_dequantized = x_quantized.dequantize()
    result_dequantized = result.dequantize()
    expected = x_dequantized + scalar_val

    # Check if they're close (allowing for quantization error)
    max_diff = (result_dequantized - expected).abs().max().item()
    tolerance = scale * 2  # Allow 2x scale for rounding errors
    assert max_diff <= tolerance, f"Max diff {max_diff} exceeds tolerance {tolerance}"


def test_operator_registration():
    """Test that the operator is properly registered."""
    # Check that the operator exists in the torchscience.ops namespace
    assert hasattr(ops, 'example')

    # Check that backward operator is registered
    assert hasattr(ops, '_example_backward')

    # Test that operator is callable
    x = torch.randn(5, 5)
    result = ops.example(x, 1.0)
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, x + 1.0)


class TestExampleOpcheck:
    """
    PT2 compliance tests using torch.library.opcheck.

    These tests validate that the example operator works correctly with
    torch.compile, torch.export, and other PyTorch 2.x APIs by checking:
    1. Schema correctness (mutations, aliasing)
    2. Autograd registration
    3. FakeTensor support (meta kernel)
    4. AOT Autograd with static shapes
    5. AOT Autograd with dynamic shapes
    """

    def get_sample_inputs(self, include_mps=False):
        """
        Generate sample inputs for opcheck testing.

        Args:
            include_mps: Whether to include MPS device tests. Set to False for
                        autograd_registration tests which don't support MPS yet.
        """
        sample_inputs = [
            # Small tensors with requires_grad for autograd testing
            (torch.randn(10, requires_grad=True, device='cpu'), 5.0),
            (torch.randn(10, requires_grad=True, device='cpu'), -2.5),

            # 2D tensors with different shapes
            (torch.randn(5, 10, requires_grad=True, device='cpu'), 1.0),
            (torch.randn(100, 100, requires_grad=True, device='cpu'), 3.5),

            # Different dtypes
            (torch.randn(20, dtype=torch.float64, requires_grad=True, device='cpu'), 2.0),
            (torch.randn(20, dtype=torch.float32, requires_grad=True, device='cpu'), -1.0),

            # Non-contiguous tensors
            (torch.randn(10, 20, requires_grad=True, device='cpu').t(), 4.0),

            # Higher dimensional tensors
            (torch.randn(2, 3, 4, 5, requires_grad=True, device='cpu'), 0.5),
        ]

        # Add CUDA tests if available
        if torch.cuda.is_available():
            sample_inputs.extend([
                (torch.randn(10, requires_grad=True, device='cuda'), 5.0),
                (torch.randn(100, 100, requires_grad=True, device='cuda'), -2.5),
                (torch.randn(20, dtype=torch.float16, requires_grad=True, device='cuda'), 1.5),
            ])

        # Add MPS tests if available
        # Note: MPS is not yet supported by PyTorch's autograd_registration_check
        if include_mps and torch.backends.mps.is_available():
            sample_inputs.extend([
                (torch.randn(10, requires_grad=True, device='mps'), 5.0),
                (torch.randn(100, 100, requires_grad=True, device='mps'), -2.5),
                (torch.randn(20, dtype=torch.float16, requires_grad=True, device='mps'), 1.5),
            ])

        return sample_inputs

    def test_opcheck_all(self):
        """
        Run all opcheck tests on the example operator.

        Note: MPS is excluded because PyTorch's autograd_registration_check
        doesn't support MPS yet.
        """
        # Don't include MPS - autograd_registration_check doesn't support it yet
        sample_inputs = self.get_sample_inputs(include_mps=False)

        for i, sample_input in enumerate(sample_inputs):
            # Run all opcheck tests
            try:
                opcheck(torch.ops.torchscience.example.default, sample_input)
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                shape = sample_input[0].shape
                pytest.fail(
                    f"opcheck failed for input {i}: "
                    f"device={device}, dtype={dtype}, shape={shape}\n"
                    f"Error: {e}"
                )

    def test_opcheck_schema(self):
        """Test schema correctness specifically."""
        sample_inputs = self.get_sample_inputs()

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(
                    torch.ops.torchscience.example.default,
                    sample_input,
                    test_utils="test_schema"
                )
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(
                    f"test_schema failed for input {i}: "
                    f"device={device}, dtype={dtype}\n"
                    f"Error: {e}"
                )

    def test_opcheck_autograd_registration(self):
        """
        Test autograd registration specifically.

        Note: MPS is excluded because PyTorch's autograd_registration_check
        doesn't support MPS yet (only CPU/CUDA/XPU).
        """
        # Don't include MPS - autograd_registration_check doesn't support it yet
        sample_inputs = self.get_sample_inputs(include_mps=False)

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(
                    torch.ops.torchscience.example.default,
                    sample_input,
                    test_utils="test_autograd_registration"
                )
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(
                    f"test_autograd_registration failed for input {i}: "
                    f"device={device}, dtype={dtype}\n"
                    f"Error: {e}"
                )

    def test_opcheck_faketensor(self):
        """Test FakeTensor support (meta kernel) specifically."""
        sample_inputs = self.get_sample_inputs()

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(
                    torch.ops.torchscience.example.default,
                    sample_input,
                    test_utils="test_faketensor"
                )
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(
                    f"test_faketensor failed for input {i}: "
                    f"device={device}, dtype={dtype}\n"
                    f"Error: {e}"
                )

    def test_opcheck_aot_autograd_static(self):
        """Test AOT Autograd with static shapes (torch.compile compatibility)."""
        sample_inputs = self.get_sample_inputs()

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(
                    torch.ops.torchscience.example.default,
                    sample_input,
                    test_utils="test_aot_dispatch_static"
                )
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(
                    f"test_aot_dispatch_static failed for input {i}: "
                    f"device={device}, dtype={dtype}\n"
                    f"Error: {e}"
                )

    def test_opcheck_aot_autograd_dynamic(self):
        """Test AOT Autograd with dynamic shapes."""
        sample_inputs = self.get_sample_inputs()

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(
                    torch.ops.torchscience.example.default,
                    sample_input,
                    test_utils="test_aot_dispatch_dynamic"
                )
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(
                    f"test_aot_dispatch_dynamic failed for input {i}: "
                    f"device={device}, dtype={dtype}\n"
                    f"Error: {e}"
                )

    def test_opcheck_cpu_only(self):
        """Quick opcheck test for CPU only (useful for CI without GPU)."""
        cpu_inputs = [
            (torch.randn(10, requires_grad=True), 5.0),
            (torch.randn(100, 100, requires_grad=True), -2.5),
            (torch.randn(20, dtype=torch.float64, requires_grad=True), 2.0),
        ]

        for sample_input in cpu_inputs:
            opcheck(torch.ops.torchscience.example.default, sample_input)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_opcheck_cuda_only(self):
        """Quick opcheck test for CUDA only."""
        cuda_inputs = [
            (torch.randn(10, requires_grad=True, device='cuda'), 5.0),
            (torch.randn(100, 100, requires_grad=True, device='cuda'), -2.5),
            (torch.randn(20, dtype=torch.float16, requires_grad=True, device='cuda'), 1.5),
        ]

        for sample_input in cuda_inputs:
            opcheck(torch.ops.torchscience.example.default, sample_input)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_opcheck_mps_only(self):
        """
        Quick opcheck test for MPS only.

        Note: We skip test_autograd_registration because PyTorch's
        autograd_registration_check doesn't support MPS yet.
        """
        mps_inputs = [
            (torch.randn(10, requires_grad=True, device='mps'), 5.0),
            (torch.randn(100, 100, requires_grad=True, device='mps'), -2.5),
            (torch.randn(20, dtype=torch.float16, requires_grad=True, device='mps'), 1.5),
        ]

        # Run all tests except autograd_registration (which doesn't support MPS)
        test_utils_for_mps = [
            "test_schema",
            "test_faketensor",
            "test_aot_dispatch_static",
            "test_aot_dispatch_dynamic",
        ]

        for sample_input in mps_inputs:
            opcheck(
                torch.ops.torchscience.example.default,
                sample_input,
                test_utils=test_utils_for_mps
            )


class TestExampleGradcheck:
    """
    Gradient correctness tests using torch.autograd.gradcheck.

    gradcheck validates that the backward pass is implemented correctly by
    comparing analytical gradients (from autograd) with numerical gradients
    (computed using finite differences).
    """

    def test_gradcheck_cpu_float64(self):
        """Test gradient correctness on CPU with float64 (most accurate)."""
        # Use float64 for better numerical precision in gradcheck
        x = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
        scalar_val = 2.5

        # Create a function that can be passed to gradcheck
        def func(x):
            return ops.example(x, scalar_val)

        # gradcheck uses finite differences to verify gradients
        assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_cpu_float32(self):
        """Test gradient correctness on CPU with float32."""
        x = torch.randn(5, 5, dtype=torch.float32, requires_grad=True)
        scalar_val = 1.5

        def func(x):
            return ops.example(x, scalar_val)

        # Use larger eps and atol for float32 due to lower precision
        assert gradcheck(func, x, eps=1e-3, atol=1e-3)

    def test_gradcheck_different_scalars(self):
        """Test gradient correctness with different scalar values."""
        scalar_values = [0.0, 1.0, -1.0, 5.5, -3.2]

        for scalar_val in scalar_values:
            x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

            def func(x):
                return ops.example(x, scalar_val)

            assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_different_shapes(self):
        """Test gradient correctness with different tensor shapes."""
        shapes = [
            (5,),           # 1D
            (3, 4),         # 2D
            (2, 3, 4),      # 3D
            (2, 2, 2, 2),   # 4D
        ]

        scalar_val = 2.0

        for shape in shapes:
            x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

            def func(x):
                return ops.example(x, scalar_val)

            assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_non_contiguous(self):
        """Test gradient correctness with non-contiguous tensors."""
        x = torch.randn(5, 10, dtype=torch.float64, requires_grad=True).t()
        assert not x.is_contiguous()

        scalar_val = 3.0

        def func(x):
            return ops.example(x, scalar_val)

        assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradgradcheck_cpu(self):
        """Test second-order gradient correctness (gradient of gradient)."""
        x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        scalar_val = 2.0

        def func(x):
            return ops.example(x, scalar_val)

        # gradgradcheck verifies second derivatives
        # For our simple addition operation, second derivative should be 0
        assert gradgradcheck(func, x, eps=1e-6, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradcheck_cuda_float64(self):
        """Test gradient correctness on CUDA with float64."""
        x = torch.randn(5, 5, dtype=torch.float64, device='cuda', requires_grad=True)
        scalar_val = 2.5

        def func(x):
            return ops.example(x, scalar_val)

        assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradcheck_cuda_float32(self):
        """Test gradient correctness on CUDA with float32."""
        x = torch.randn(5, 5, dtype=torch.float32, device='cuda', requires_grad=True)
        scalar_val = 1.5

        def func(x):
            return ops.example(x, scalar_val)

        assert gradcheck(func, x, eps=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradgradcheck_cuda(self):
        """Test second-order gradient correctness on CUDA."""
        x = torch.randn(3, 3, dtype=torch.float64, device='cuda', requires_grad=True)
        scalar_val = 2.0

        def func(x):
            return ops.example(x, scalar_val)

        assert gradgradcheck(func, x, eps=1e-6, atol=1e-4)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_gradcheck_mps_float32(self):
        """
        Test gradient correctness on MPS with float32.

        Note: MPS doesn't support float64, so we use float32 with relaxed tolerances.
        """
        x = torch.randn(5, 5, dtype=torch.float32, device='mps', requires_grad=True)
        scalar_val = 1.5

        def func(x):
            return ops.example(x, scalar_val)

        # MPS may have slightly different numerical behavior, so use relaxed tolerances
        assert gradcheck(func, x, eps=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_gradcheck_mps_different_shapes(self):
        """Test gradient correctness on MPS with different shapes."""
        shapes = [(5,), (3, 4), (2, 3, 4)]
        scalar_val = 2.0

        for shape in shapes:
            x = torch.randn(*shape, dtype=torch.float32, device='mps', requires_grad=True)

            def func(x):
                return ops.example(x, scalar_val)

            assert gradcheck(func, x, eps=1e-3, atol=1e-3)

    def test_gradcheck_zero_scalar(self):
        """Test gradient correctness when scalar is zero (identity operation)."""
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

        def func(x):
            return ops.example(x, 0.0)

        assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_negative_scalar(self):
        """Test gradient correctness with negative scalar."""
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

        def func(x):
            return ops.example(x, -5.0)

        assert gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_chain_rule(self):
        """Test gradient correctness in a chain of operations."""
        x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

        def func(x):
            # Chain multiple operations together
            y = ops.example(x, 2.0)
            z = y * 3.0
            return ops.example(z, 1.0)

        assert gradcheck(func, x, eps=1e-6, atol=1e-4)
