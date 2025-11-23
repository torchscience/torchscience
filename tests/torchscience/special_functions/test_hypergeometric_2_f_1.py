import sys

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.library import opcheck

from torchscience.special_functions import hypergeometric_2_f_1

# Try to import mpmath for reference validation
try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


class TestHypergeometric2F1Operator:
    """Test suite for the hypergeometric_2_f_1 operator (Gaussian hypergeometric function)."""

    @pytest.fixture(params=["cpu"])
    def device(self, request):
        """Parametrize tests across available devices."""
        return request.param

    @pytest.fixture(params=[torch.float32, torch.float64])
    def dtype(self, request):
        """Parametrize tests across floating-point data types."""
        return request.param

    def test_basic_functionality(self, device, dtype):
        """Test basic functionality with simple inputs."""
        a = torch.tensor([1.0], dtype=dtype, device=device)
        b = torch.tensor([2.0], dtype=dtype, device=device)
        c = torch.tensor([3.0], dtype=dtype, device=device)
        z = torch.tensor([0.5], dtype=dtype, device=device)

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.shape == (1,)
        assert result.dtype == dtype
        assert result.device.type == device

    def test_shape_preservation(self, device):
        """Test that output shape matches broadcasted input shapes."""
        # All same shape
        a = torch.randn(10, 5, device=device)
        b = torch.randn(10, 5, device=device)
        c = torch.randn(10, 5, device=device)
        z = torch.randn(10, 5, device=device)

        result = hypergeometric_2_f_1(a, b, c, z)
        assert result.shape == (10, 5)

    def test_broadcasting(self, device):
        """Test broadcasting across different input shapes."""
        # Test various broadcasting scenarios
        test_cases = [
            # (a_shape, b_shape, c_shape, z_shape, expected_shape)
            ((1,), (5,), (1,), (5,), (5,)),
            ((5, 1), (1, 3), (5, 3), (5, 3), (5, 3)),
            ((10,), (10,), (10,), (1,), (10,)),
            ((2, 1, 4), (1, 3, 1), (1, 1, 1), (2, 3, 4), (2, 3, 4)),
        ]

        for a_shape, b_shape, c_shape, z_shape, expected_shape in test_cases:
            a = torch.randn(*a_shape, device=device)
            b = torch.randn(*b_shape, device=device)
            c = torch.randn(*c_shape, device=device)
            z = torch.randn(*z_shape, device=device) * 0.5  # Keep |z| < 1

            result = hypergeometric_2_f_1(a, b, c, z)
            assert result.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {result.shape}"

    def test_empty_tensor(self, device):
        """Test with empty tensors."""
        a = torch.empty(0, dtype=torch.float32, device=device)
        b = torch.empty(0, dtype=torch.float32, device=device)
        c = torch.empty(0, dtype=torch.float32, device=device)
        z = torch.empty(0, dtype=torch.float32, device=device)

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.shape == (0,)
        assert result.numel() == 0

    def test_scalar_tensor(self, device):
        """Test with scalar tensors (0-dimensional)."""
        a = torch.tensor(1.0, device=device)
        b = torch.tensor(2.0, device=device)
        c = torch.tensor(3.0, device=device)
        z = torch.tensor(0.5, device=device)

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.shape == ()
        assert result.ndim == 0

    def test_special_value_z_zero(self, device):
        """Test special case: z=0 should give 1."""
        a = torch.rand(5, device=device)
        b = torch.rand(5, device=device)
        c = torch.rand(5, device=device) + 1.0  # Ensure c > 0
        z = torch.zeros(5, device=device)

        result = hypergeometric_2_f_1(a, b, c, z)

        # ₂F₁(a,b;c;0) = 1
        assert torch.allclose(result, torch.ones(5, device=device), atol=1e-6)

    def test_different_shapes(self, device):
        """Test with various tensor shapes."""
        shapes = [
            (5,),  # 1D
            (5, 5),  # 2D square
            (3, 7),  # 2D rectangular
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        for shape in shapes:
            a = torch.rand(*shape, device=device)
            b = torch.rand(*shape, device=device)
            c = torch.rand(*shape, device=device) + 1.0
            z = torch.rand(*shape, device=device) * 0.5

            result = hypergeometric_2_f_1(a, b, c, z)
            assert result.shape == shape

    def test_complex_dtypes(self, device):
        """Test with complex data types."""
        complex_dtypes = [torch.complex64, torch.complex128]

        for dtype in complex_dtypes:
            a = torch.randn(5, dtype=dtype, device=device)
            b = torch.randn(5, dtype=dtype, device=device)
            c = torch.randn(5, dtype=dtype, device=device) + 1.0
            z = torch.randn(5, dtype=dtype, device=device) * 0.3

            result = hypergeometric_2_f_1(a, b, c, z)

            assert result.dtype == dtype
            assert result.shape == (5,)

    def test_contiguous_and_non_contiguous(self, device):
        """Test with both contiguous and non-contiguous tensors."""
        a = torch.randn(10, 20, device=device)
        b = torch.randn(10, 20, device=device)
        c = torch.randn(10, 20, device=device) + 1.0
        z = torch.randn(10, 20, device=device) * 0.5

        # Test contiguous
        assert a.is_contiguous()
        result_contiguous = hypergeometric_2_f_1(a, b, c, z)
        assert result_contiguous.shape == (10, 20)

        # Test non-contiguous (transposed)
        a_t, b_t, c_t, z_t = a.t(), b.t(), c.t(), z.t()
        assert not a_t.is_contiguous()
        result_non_contiguous = hypergeometric_2_f_1(a_t, b_t, c_t, z_t)
        assert result_non_contiguous.shape == (20, 10)

    @pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath not available")
    def test_reference_validation_mpmath(self, device):
        """Validate results against mpmath reference implementation."""
        # Use mpmath with high precision
        mpmath.mp.dps = 50  # 50 decimal places

        # Test specific values
        test_cases = [
            (1.0, 2.0, 3.0, 0.1),
            (0.5, 1.5, 2.5, 0.3),
            (1.0, 1.0, 2.0, 0.5),
            (2.0, 3.0, 4.0, -0.5),
            (0.5, 0.5, 1.5, 0.9),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([a_val], dtype=torch.float64, device=device)
            b = torch.tensor([b_val], dtype=torch.float64, device=device)
            c = torch.tensor([c_val], dtype=torch.float64, device=device)
            z = torch.tensor([z_val], dtype=torch.float64, device=device)

            result = hypergeometric_2_f_1(a, b, c, z)

            # Compute reference with mpmath
            expected = float(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            # Create expected tensor with same dtype as result
            assert torch.allclose(result, torch.tensor([expected], dtype=torch.float64, device=device),
                                  rtol=1e-10, atol=1e-12), \
                f"Failed for (a={a_val}, b={b_val}, c={c_val}, z={z_val}): " \
                f"got {result.item()}, expected {expected}"

    @pytest.mark.skipif(not MPMATH_AVAILABLE, reason="mpmath not available")
    def test_reference_validation_complex(self, device):
        """Validate complex results against mpmath."""
        if device != "cpu":
            pytest.skip("Complex mpmath validation only on CPU")

        mpmath.mp.dps = 30

        # Test with complex z
        test_cases = [
            (1.0, 2.0, 3.0, 0.5 + 0.5j),
            (0.5, 1.5, 2.5, 0.3 + 0.2j),
            (1.0, 1.0, 2.0, -0.3 + 0.4j),
        ]

        for a_val, b_val, c_val, z_val in test_cases:
            a = torch.tensor([complex(a_val)], dtype=torch.complex128, device=device)
            b = torch.tensor([complex(b_val)], dtype=torch.complex128, device=device)
            c = torch.tensor([complex(c_val)], dtype=torch.complex128, device=device)
            z = torch.tensor([z_val], dtype=torch.complex128, device=device)

            result = hypergeometric_2_f_1(a, b, c, z)

            # Compute reference with mpmath
            expected = complex(mpmath.hyp2f1(a_val, b_val, c_val, z_val))

            # Create expected tensor with same dtype as result
            assert torch.allclose(result, torch.tensor([expected], dtype=torch.complex128, device=device),
                                  rtol=1e-8, atol=1e-10), \
                f"Failed for (a={a_val}, b={b_val}, c={c_val}, z={z_val})"

    def test_autograd_basic(self):
        """Test autograd functionality."""
        a = torch.randn(5, requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        # Create leaf tensors with constrained values
        # uniform_() must be called before enabling requires_grad
        c = torch.empty(5).uniform_(0.5, 1.5).requires_grad_(True)
        z = torch.empty(5).uniform_(-0.5, 0.5).requires_grad_(True)

        result = hypergeometric_2_f_1(a, b, c, z)

        # Compute loss and backward
        loss = result.sum()
        loss.backward()

        # Check that gradients are computed
        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None
        assert z.grad is not None

    def test_autograd_chain(self):
        """Test autograd with chain of operations."""
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        # Create leaf tensors with constrained values
        c = torch.empty(3, 3).uniform_(0.5, 1.5).requires_grad_(True)
        z = torch.empty(3, 3).uniform_(-0.3, 0.3).requires_grad_(True)

        # Chain operations
        y = hypergeometric_2_f_1(a, b, c, z)
        w = y * 2.0
        loss = w.sum()

        loss.backward()

        # Gradients should be computed through the chain
        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None
        assert z.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test CUDA backend."""
        a = torch.rand(50, 50, device="cuda")
        b = torch.rand(50, 50, device="cuda")
        c = torch.rand(50, 50, device="cuda") + 1.0
        z = torch.rand(50, 50, device="cuda") * 0.5

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.device.type == "cuda"
        assert result.shape == (50, 50)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_complex(self):
        """Test CUDA backend with complex types."""
        a = torch.randn(10, 10, dtype=torch.complex64, device="cuda")
        b = torch.randn(10, 10, dtype=torch.complex64, device="cuda")
        c = torch.randn(10, 10, dtype=torch.complex64, device="cuda") + 1.0
        z = torch.randn(10, 10, dtype=torch.complex64, device="cuda") * 0.3

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.dtype == torch.complex64
        assert result.device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps(self):
        """Test MPS backend (Apple Silicon)."""
        a = torch.rand(50, 50, device="mps")
        b = torch.rand(50, 50, device="mps")
        c = torch.rand(50, 50, device="mps") + 1.0
        z = torch.rand(50, 50, device="mps") * 0.5

        result = hypergeometric_2_f_1(a, b, c, z)

        assert result.device.type == "mps"
        assert result.shape == (50, 50)

    def test_inplace_safety(self, device):
        """Test that the operator doesn't modify inputs in-place."""
        a = torch.randn(5, 5, device=device)
        b = torch.randn(5, 5, device=device)
        c = torch.randn(5, 5, device=device) + 1.0
        z = torch.randn(5, 5, device=device) * 0.5

        a_orig = a.clone()
        b_orig = b.clone()
        c_orig = c.clone()
        z_orig = z.clone()

        result = hypergeometric_2_f_1(a, b, c, z)

        # Inputs should remain unchanged
        assert torch.allclose(a, a_orig)
        assert torch.allclose(b, b_orig)
        assert torch.allclose(c, c_orig)
        assert torch.allclose(z, z_orig)

    def test_deterministic(self, device):
        """Test that the operator is deterministic."""
        # Use constrained inputs to avoid NaN/inf values that make comparison difficult
        # |z| < 0.8 ensures convergence, c > 0.5 avoids division by zero
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        c = torch.rand(10, 10, device=device) + 0.5  # (0.5, 1.5)
        z = torch.rand(10, 10, device=device) * 1.6 - 0.8  # (-0.8, 0.8)

        result1 = hypergeometric_2_f_1(a, b, c, z)
        result2 = hypergeometric_2_f_1(a, b, c, z)

        assert torch.allclose(result1, result2)


def test_operator_registration():
    """Test that the operator is properly registered and accessible."""
    from torchscience import special_functions

    # Should be accessible via special_functions module
    assert hasattr(special_functions, "hypergeometric_2_f_1")

    # Test that operator is callable
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])
    z = torch.tensor([0.5])

    result = hypergeometric_2_f_1(a, b, c, z)
    assert result is not None
    assert isinstance(result, torch.Tensor)


def test_python_wrapper_direct_import():
    """Test that the Python wrapper can be imported and used directly."""
    # Test direct import works
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.tensor([3.0])
    z = torch.tensor([0.5])

    # Call using direct import
    result = hypergeometric_2_f_1(a, b, c, z)
    assert result is not None
    assert isinstance(result, torch.Tensor)

    # Verify it's deterministic
    result2 = hypergeometric_2_f_1(a, b, c, z)
    assert torch.allclose(result, result2)


def test_quantized_cpu():
    """Test quantized CPU backend."""
    # Create regular float tensors
    a_float = torch.rand(10, 10)
    b_float = torch.rand(10, 10)
    c_float = torch.rand(10, 10) + 1.0
    z_float = torch.rand(10, 10) * 0.5

    # Quantize inputs
    scale = 0.1
    zero_point = 0

    z_quantized = torch.quantize_per_tensor(z_float, scale, zero_point, torch.qint8)

    # Apply operator
    result = hypergeometric_2_f_1(a_float, b_float, c_float, z_quantized)

    # Check that result preserves quantization scheme
    assert result.is_quantized
    assert result.q_scale() == z_quantized.q_scale()
    assert result.q_zero_point() == z_quantized.q_zero_point()


class TestHypergeometric2F1Opcheck:
    """PT2 compliance tests using torch.library.opcheck."""

    def get_sample_inputs(self, include_mps=False):
        """Generate sample inputs for opcheck testing.

        Uses constrained inputs to avoid NaN/inf values:
        - c in (0.5, 1.5) to avoid division by zero
        - z in (-0.8, 0.8) to ensure convergence (|z| < 1)
        """
        sample_inputs = [
            # Basic cases
            (
                torch.randn(5, requires_grad=True),
                torch.randn(5, requires_grad=True),
                torch.rand(5, requires_grad=True) + 0.5,  # (0.5, 1.5)
                torch.rand(5, requires_grad=True) * 1.6 - 0.8,  # (-0.8, 0.8)
            ),
            # Different shapes (broadcasting)
            (
                torch.randn(10, 1, requires_grad=True),
                torch.randn(1, 10, requires_grad=True),
                torch.rand(10, 10, requires_grad=True) + 0.5,  # (0.5, 1.5)
                torch.rand(10, 10, requires_grad=True) * 1.6 - 0.8,  # (-0.8, 0.8)
            ),
            # Different dtypes
            (
                torch.randn(5, dtype=torch.float64, requires_grad=True),
                torch.randn(5, dtype=torch.float64, requires_grad=True),
                torch.rand(5, dtype=torch.float64, requires_grad=True) + 0.5,  # (0.5, 1.5)
                torch.rand(5, dtype=torch.float64, requires_grad=True) * 1.6 - 0.8,  # (-0.8, 0.8)
            ),
        ]

        # Add CUDA tests if available
        if torch.cuda.is_available():
            sample_inputs.extend([
                (
                    torch.randn(5, device="cuda", requires_grad=True),
                    torch.randn(5, device="cuda", requires_grad=True),
                    torch.rand(5, device="cuda", requires_grad=True) + 0.5,  # (0.5, 1.5)
                    torch.rand(5, device="cuda", requires_grad=True) * 1.6 - 0.8,  # (-0.8, 0.8)
                ),
            ])

        # Add MPS tests if available
        if include_mps and torch.backends.mps.is_available():
            sample_inputs.extend([
                (
                    torch.randn(5, device="mps", requires_grad=True),
                    torch.randn(5, device="mps", requires_grad=True),
                    torch.rand(5, device="mps", requires_grad=True) + 0.5,  # (0.5, 1.5)
                    torch.rand(5, device="mps", requires_grad=True) * 1.6 - 0.8,  # (-0.8, 0.8)
                ),
            ])

        return sample_inputs

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 14),
        reason="PyTorch opcheck has Python 3.14 compatibility issues",
    )
    def test_opcheck_all(self):
        """Run all opcheck tests on the hypergeometric_2_f_1 operator."""
        sample_inputs = self.get_sample_inputs(include_mps=False)

        for i, sample_input in enumerate(sample_inputs):
            try:
                opcheck(torch.ops.torchscience.hypergeometric_2_f_1.default, sample_input)
            except Exception as e:
                device = sample_input[0].device
                dtype = sample_input[0].dtype
                pytest.fail(f"opcheck failed for input {i}: device={device}, dtype={dtype}\nError: {e}")


class TestHypergeometric2F1Gradcheck:
    """Gradient correctness tests using torch.autograd.gradcheck."""

    def test_gradcheck_cpu_float64(self):
        """Test gradient correctness on CPU with float64."""
        a = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        c = torch.randn(3, 3, dtype=torch.float64, requires_grad=True) + 1.0
        z = torch.randn(3, 3, dtype=torch.float64, requires_grad=True) * 0.3

        def func(a, b, c, z):
            return hypergeometric_2_f_1(a, b, c, z)

        # Use larger eps due to finite differences in backward
        assert gradcheck(func, (a, b, c, z), eps=1e-3, atol=1e-2)

    def test_gradcheck_single_param(self):
        """Test gradient correctness for z parameter with fixed a, b, c."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

        def func(z):
            return hypergeometric_2_f_1(a, b, c, z)

        assert gradcheck(func, z, eps=1e-4, atol=1e-3)


class TestHypergeometric2F1Sparse:
    """Tests for sparse tensor support."""

    def test_sparse_cpu_basic(self):
        """Test basic sparse CPU operation."""
        # Create sparse tensors
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        a_values = torch.tensor([1.0, 2.0, 3.0])
        b_values = torch.tensor([2.0, 3.0, 4.0])
        c_values = torch.tensor([3.0, 4.0, 5.0])
        z_values = torch.tensor([0.1, 0.2, 0.3])

        a_sparse = torch.sparse_coo_tensor(indices, a_values, (3, 3))
        b_sparse = torch.sparse_coo_tensor(indices, b_values, (3, 3))
        c_sparse = torch.sparse_coo_tensor(indices, c_values, (3, 3))
        z_sparse = torch.sparse_coo_tensor(indices, z_values, (3, 3))

        result = hypergeometric_2_f_1(a_sparse, b_sparse, c_sparse, z_sparse)

        # Check result is sparse
        assert result.is_sparse

        # Check indices unchanged
        assert torch.equal(result._indices(), indices)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sparse_cuda_basic(self):
        """Test basic sparse CUDA operation."""
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device="cuda")
        z_values = torch.tensor([0.1, 0.2, 0.3], device="cuda")

        z_sparse = torch.sparse_coo_tensor(indices, z_values, (3, 3))

        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = torch.tensor([2.0, 3.0, 4.0], device="cuda")
        c = torch.tensor([3.0, 4.0, 5.0], device="cuda")

        result = hypergeometric_2_f_1(a, b, c, z_sparse)

        assert result.is_sparse
        assert result.is_cuda
