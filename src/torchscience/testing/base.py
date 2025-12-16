from abc import ABC, abstractmethod
from typing import Tuple

import pytest
import torch
import torch.testing

from .descriptors import InputSpec, OperatorDescriptor
from .sympy_utils import SymPyReference

# =============================================================================
# Test Mixins - PyTorch Features
# =============================================================================


class AutogradTestsMixin:
    """Mixin providing autograd tests for operators."""

    descriptor: OperatorDescriptor

    def _make_gradcheck_inputs(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...] = (3,),
    ) -> Tuple[torch.Tensor, ...]:
        """Generate inputs suitable for gradient checking."""
        inputs = []
        for spec in self.descriptor.input_specs:
            tensor = self._make_input_for_spec(spec, dtype, "cpu", shape)
            if (
                spec.supports_grad
                and dtype.is_floating_point
                or dtype.is_complex
            ):
                tensor = tensor.requires_grad_(True)
            inputs.append(tensor)
        return tuple(inputs)

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gradcheck_real(self, dtype: torch.dtype):
        """Test gradient correctness for real dtypes using torch.autograd.gradcheck."""
        if "test_gradcheck_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradcheck(
            func,
            inputs,
            eps=tol.gradcheck_eps,
            atol=tol.gradcheck_atol,
            rtol=tol.gradcheck_rtol,
        )

    @pytest.mark.parametrize("dtype", [torch.complex128])
    def test_gradcheck_complex(self, dtype: torch.dtype):
        """Test gradient correctness for complex dtypes."""
        if "test_gradcheck_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradcheck(
            func,
            inputs,
            eps=tol.gradcheck_eps,
            atol=tol.gradcheck_atol * 10,  # Relaxed for complex
            rtol=tol.gradcheck_rtol * 10,
        )

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gradgradcheck_real(self, dtype: torch.dtype):
        """Test second-order gradient correctness for real dtypes."""
        if "test_gradgradcheck_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradgradcheck(
            func,
            inputs,
            eps=tol.gradgradcheck_eps,
            atol=tol.gradgradcheck_atol,
            rtol=tol.gradgradcheck_rtol,
        )

    @pytest.mark.parametrize("dtype", [torch.complex128])
    def test_gradgradcheck_complex(self, dtype: torch.dtype):
        """Test second-order gradient correctness for complex dtypes."""
        if "test_gradgradcheck_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_gradcheck_inputs(dtype)
        tol = self.descriptor.tolerances

        def func(*args):
            return self.descriptor.func(*args)

        assert torch.autograd.gradgradcheck(
            func,
            inputs,
            eps=tol.gradgradcheck_eps,
            atol=tol.gradgradcheck_atol * 10,  # Relaxed for complex
            rtol=tol.gradgradcheck_rtol * 10,
        )


class DeviceTestsMixin:
    """Mixin providing device (CPU/CUDA) tests."""

    descriptor: OperatorDescriptor

    def test_cpu_device(self):
        """Test operator works on CPU."""
        if "test_cpu_device" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(device="cpu")
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cpu"
        assert torch.isfinite(result).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test operator works on CUDA."""
        if "test_cuda_device" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(device="cuda")
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cuda"

        # Compare against CPU
        cpu_inputs = tuple(t.cpu() for t in inputs)
        expected = self.descriptor.func(*cpu_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(
            result.cpu(), expected, rtol=rtol, atol=atol
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_complex(self):
        """Test complex dtypes on CUDA."""
        if "test_cuda_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(
            dtype=torch.complex128, device="cuda"
        )
        result = self.descriptor.func(*inputs)
        assert result.device.type == "cuda"
        assert result.dtype == torch.complex128


class DtypeTestsMixin:
    """Mixin providing dtype handling tests."""

    descriptor: OperatorDescriptor

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_dtypes(self, dtype: torch.dtype):
        """Test operator with real floating-point dtypes."""
        if "test_real_dtypes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_dtypes(self, dtype: torch.dtype):
        """Test operator with complex dtypes."""
        if "test_complex_dtypes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        if "test_dtype_preservation" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for dtype in [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]:
            inputs = self._make_standard_inputs(dtype=dtype)
            result = self.descriptor.func(*inputs)
            assert result.dtype == dtype, (
                f"Expected {dtype}, got {result.dtype}"
            )


class LowPrecisionTestsMixin:
    """Mixin providing float16/bfloat16 tests."""

    descriptor: OperatorDescriptor

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, dtype: torch.dtype):
        """Test forward pass with low-precision dtypes."""
        if "test_low_precision_forward" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

        # Compare against float32 reference
        fp32_inputs = tuple(t.to(torch.float32) for t in inputs)
        expected = self.descriptor.func(*fp32_inputs)
        rtol, atol = self.descriptor.tolerances.get_tolerances(dtype)
        torch.testing.assert_close(
            result.to(torch.float32), expected, rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_dtype_preservation(self, dtype: torch.dtype):
        """Test that low-precision dtype is preserved."""
        if (
            "test_low_precision_dtype_preservation"
            in self.descriptor.skip_tests
        ):
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype)
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_cuda(self, dtype: torch.dtype):
        """Test low-precision on CUDA."""
        if "test_low_precision_cuda" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=dtype, device="cuda")
        result = self.descriptor.func(*inputs)
        assert result.dtype == dtype
        assert result.device.type == "cuda"


class BroadcastingTestsMixin:
    """Mixin providing broadcasting tests."""

    descriptor: OperatorDescriptor

    def test_broadcast_scalar_with_tensor(self):
        """Test broadcasting scalar with tensor."""
        if "test_broadcast_scalar_with_tensor" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires binary operator")

        # Create scalar and tensor inputs
        spec0 = self.descriptor.input_specs[0]
        spec1 = self.descriptor.input_specs[1]

        scalar = self._make_input_for_spec(spec0, torch.float64, "cpu", ())
        tensor = self._make_input_for_spec(spec1, torch.float64, "cpu", (5,))

        result = self.descriptor.func(scalar, tensor)
        assert result.shape == (5,)

    def test_broadcast_different_shapes(self):
        """Test broadcasting with different shapes."""
        if "test_broadcast_different_shapes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires binary operator")

        spec0 = self.descriptor.input_specs[0]
        spec1 = self.descriptor.input_specs[1]

        t1 = self._make_input_for_spec(spec0, torch.float64, "cpu", (3, 1))
        t2 = self._make_input_for_spec(spec1, torch.float64, "cpu", (1, 4))

        result = self.descriptor.func(t1, t2)
        assert result.shape == (3, 4)


class TorchCompileTestsMixin:
    """Mixin providing torch.compile tests."""

    descriptor: OperatorDescriptor

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_smoke(self):
        """Smoke test for torch.compile compatibility."""
        if "test_compile_smoke" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        @torch.compile
        def compiled_func(*args):
            return self.descriptor.func(*args)

        inputs = self._make_standard_inputs()
        result = compiled_func(*inputs)
        expected = self.descriptor.func(*inputs)

        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


class VmapTestsMixin:
    """Mixin providing vmap tests."""

    descriptor: OperatorDescriptor

    def test_vmap_over_batch(self):
        """Test vmap over batch dimension."""
        if "test_vmap_over_batch" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(shape=(5, 3))

        # vmap over first dimension
        vmapped = torch.vmap(
            self.descriptor.func, in_dims=tuple(0 for _ in inputs)
        )
        result = vmapped(*inputs)

        # Compare with manual loop
        expected = torch.stack(
            [self.descriptor.func(*[t[i] for t in inputs]) for i in range(5)]
        )

        rtol, atol = self.descriptor.tolerances.get_tolerances(inputs[0].dtype)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


class SparseTestsMixin:
    """Mixin providing sparse tensor tests."""

    descriptor: OperatorDescriptor

    def test_sparse_coo_basic(self):
        """Test basic sparse COO tensor support."""
        if "test_sparse_coo_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_coo:
            pytest.skip("Operator does not support sparse COO")

        if self.descriptor.arity != 1:
            pytest.skip("Sparse test only for unary operators")

        # Create sparse COO tensor
        indices = torch.tensor([[0, 1, 3], [1, 2, 0]])
        values = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        sparse = torch.sparse_coo_tensor(indices, values, (4, 3))

        result = self.descriptor.func(sparse)

        # Verify sparsity is preserved
        assert result.is_sparse
        assert result.shape == sparse.shape

        # Compare with dense computation
        dense_result = self.descriptor.func(sparse.to_dense())
        rtol, atol = self.descriptor.tolerances.get_tolerances(torch.float64)
        torch.testing.assert_close(
            result.to_dense(), dense_result, rtol=rtol, atol=atol
        )

    def test_sparse_csr_basic(self):
        """Test basic sparse CSR tensor support."""
        if "test_sparse_csr_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_sparse_csr:
            pytest.skip("Operator does not support sparse CSR")

        if self.descriptor.arity != 1:
            pytest.skip("Sparse test only for unary operators")

        # Create sparse CSR tensor
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([1, 2, 0])
        values = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        sparse = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, (3, 3)
        )

        result = self.descriptor.func(sparse)

        # Verify sparsity is preserved
        assert result.layout == torch.sparse_csr
        assert result.shape == sparse.shape


class QuantizedTestsMixin:
    """Mixin providing quantized tensor tests."""

    descriptor: OperatorDescriptor

    @pytest.mark.parametrize("qtype", [torch.quint8, torch.qint8])
    def test_quantized_basic(self, qtype: torch.dtype):
        """Test basic quantized tensor support."""
        if "test_quantized_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_quantized:
            pytest.skip("Operator does not support quantized tensors")

        if self.descriptor.arity != 1:
            pytest.skip("Quantized test only for unary operators")

        # Create quantized tensor
        scale = 0.1
        zero_point = 0
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        qx = torch.quantize_per_tensor(x, scale, zero_point, qtype)

        result = self.descriptor.func(qx)
        assert result.is_quantized


class MetaTensorTestsMixin:
    """Mixin providing meta tensor tests."""

    descriptor: OperatorDescriptor

    def test_meta_tensor_shape_inference(self):
        """Test shape inference with meta tensors."""
        if "test_meta_tensor_shape_inference" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_meta:
            pytest.skip("Operator does not support meta tensors")

        inputs = []
        for spec in self.descriptor.input_specs:
            inputs.append(torch.empty(5, dtype=torch.float64, device="meta"))

        result = self.descriptor.func(*inputs)
        assert result.device.type == "meta"
        assert result.shape == (5,)

    def test_meta_tensor_large_shape(self):
        """Test meta tensors with large shapes (no memory allocation)."""
        if "test_meta_tensor_large_shape" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_meta:
            pytest.skip("Operator does not support meta tensors")

        inputs = []
        for spec in self.descriptor.input_specs:
            inputs.append(
                torch.empty(10000, 10000, dtype=torch.float64, device="meta")
            )

        result = self.descriptor.func(*inputs)
        assert result.device.type == "meta"
        assert result.shape == (10000, 10000)


class AutocastTestsMixin:
    """Mixin providing autocast (AMP) tests."""

    descriptor: OperatorDescriptor

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda_float16(self):
        """Test CUDA autocast with float16."""
        if "test_autocast_cuda_float16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.float16

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda_bfloat16(self):
        """Test CUDA autocast with bfloat16."""
        if "test_autocast_cuda_bfloat16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.bfloat16

    def test_autocast_cpu_bfloat16(self):
        """Test CPU autocast with bfloat16."""
        if "test_autocast_cpu_bfloat16" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        inputs = self._make_standard_inputs(dtype=torch.float32, device="cpu")

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = self.descriptor.func(*inputs)

        assert result.dtype == torch.bfloat16


class NaNInfTestsMixin:
    """Mixin providing NaN/Inf propagation tests."""

    descriptor: OperatorDescriptor

    def test_nan_propagation(self):
        """Test that NaN inputs produce NaN outputs."""
        if "test_nan_propagation" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        spec = self.descriptor.input_specs[0]
        x = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)

        if self.descriptor.arity == 1:
            result = self.descriptor.func(x)
        else:
            other = self._make_input_for_spec(
                self.descriptor.input_specs[1], torch.float64, "cpu", (3,)
            )
            result = self.descriptor.func(x, other)

        assert torch.isnan(result[1])

    def test_inf_handling(self):
        """Test behavior with infinite inputs."""
        if "test_inf_handling" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        spec = self.descriptor.input_specs[0]
        x = torch.tensor(
            [1.0, float("inf"), float("-inf")], dtype=torch.float64
        )

        if self.descriptor.arity == 1:
            result = self.descriptor.func(x)
        else:
            other = self._make_input_for_spec(
                self.descriptor.input_specs[1], torch.float64, "cpu", (3,)
            )
            result = self.descriptor.func(x, other)

        # Just verify it doesn't crash - specific behavior depends on function
        assert result.shape == (3,)


# =============================================================================
# Test Mixins - Mathematical Properties
# =============================================================================


class SymPyReferenceTestsMixin:
    """Mixin providing SymPy reference verification tests."""

    descriptor: OperatorDescriptor

    def test_sympy_reference_real(self):
        """Verify real values match SymPy reference."""
        if "test_sympy_reference_real" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.sympy_func is None:
            pytest.skip("No SymPy function provided")

        inputs = self._make_standard_inputs(dtype=torch.float64, shape=(5,))
        result = self.descriptor.func(*inputs)

        reference = SymPyReference(self.descriptor.sympy_func)
        expected = reference.to_torch(*inputs, dtype=result.dtype)

        rtol = self.descriptor.tolerances.sympy_rtol
        atol = self.descriptor.tolerances.sympy_atol
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_sympy_reference_complex(self):
        """Verify complex values match SymPy reference."""
        if "test_sympy_reference_complex" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.sympy_func is None:
            pytest.skip("No SymPy function provided")

        inputs = self._make_standard_inputs(dtype=torch.complex128, shape=(5,))
        result = self.descriptor.func(*inputs)

        reference = SymPyReference(self.descriptor.sympy_func)
        expected = reference.to_torch(*inputs, dtype=result.dtype)

        rtol = (
            self.descriptor.tolerances.sympy_rtol * 10
        )  # Relaxed for complex
        atol = self.descriptor.tolerances.sympy_atol * 10
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


class RecurrenceTestsMixin:
    """Mixin providing recurrence relation tests."""

    descriptor: OperatorDescriptor

    def test_recurrence_relations(self):
        """Test all configured recurrence relations."""
        if "test_recurrence_relations" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.recurrence_relations:
            assert spec.check_fn(self.descriptor.func), (
                f"Recurrence '{spec.name}' failed: {spec.description}"
            )


class IdentityTestsMixin:
    """Mixin providing functional identity tests."""

    descriptor: OperatorDescriptor

    def test_functional_identities(self):
        """Test all configured functional identities."""
        if "test_functional_identities" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.functional_identities:
            left, right = spec.identity_fn(self.descriptor.func)
            (
                torch.testing.assert_close(
                    left,
                    right,
                    rtol=spec.rtol,
                    atol=spec.atol,
                ),
                f"Identity '{spec.name}' failed: {spec.description}",
            )


class SpecialValueTestsMixin:
    """Mixin providing special value tests."""

    descriptor: OperatorDescriptor

    def test_special_values(self):
        """Test all configured special values."""
        if "test_special_values" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        for spec in self.descriptor.special_values:
            inputs = tuple(
                torch.tensor([v], dtype=torch.float64) for v in spec.inputs
            )
            result = self.descriptor.func(*inputs)
            expected = torch.tensor([spec.expected], dtype=torch.float64)
            (
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=spec.rtol,
                    atol=spec.atol,
                ),
                f"Special value {spec.inputs} -> {spec.expected} failed: {spec.description}",
            )


class SingularityTestsMixin:
    """Mixin providing singularity behavior tests."""

    descriptor: OperatorDescriptor

    def test_pole_behavior(self):
        """Test behavior at poles."""
        if "test_pole_behavior" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        pole_specs = [
            s for s in self.descriptor.singularities if s.type == "pole"
        ]
        if not pole_specs:
            pytest.skip("No pole singularities defined")

        for spec in pole_specs:
            for loc in list(spec.locations())[:10]:  # Test first 10 poles
                x = torch.tensor([loc], dtype=torch.float64)
                if self.descriptor.arity == 1:
                    result = self.descriptor.func(x)
                else:
                    other = self._make_input_for_spec(
                        self.descriptor.input_specs[1],
                        torch.float64,
                        "cpu",
                        (1,),
                    )
                    result = self.descriptor.func(x, other)

                if spec.expected_behavior == "inf":
                    assert (
                        torch.isinf(result).all() or torch.isnan(result).all()
                    ), f"Expected inf/nan at pole {loc}, got {result}"


# =============================================================================
# Base Test Case Classes
# =============================================================================


class OpTestCase(
    ABC,
    AutogradTestsMixin,
    DeviceTestsMixin,
    DtypeTestsMixin,
    LowPrecisionTestsMixin,
    BroadcastingTestsMixin,
    TorchCompileTestsMixin,
    VmapTestsMixin,
    SparseTestsMixin,
    QuantizedTestsMixin,
    MetaTensorTestsMixin,
    AutocastTestsMixin,
    NaNInfTestsMixin,
    SymPyReferenceTestsMixin,
    RecurrenceTestsMixin,
    IdentityTestsMixin,
    SpecialValueTestsMixin,
    SingularityTestsMixin,
):
    """Base test case for PyTorch operators."""

    @property
    @abstractmethod
    def descriptor(self) -> OperatorDescriptor:
        """Return the operator descriptor."""
        ...

    def _make_standard_inputs(
        self,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        shape: Tuple[int, ...] = (5,),
    ) -> Tuple[torch.Tensor, ...]:
        """Generate standard test inputs based on descriptor."""
        inputs = []
        for spec in self.descriptor.input_specs:
            tensor = self._make_input_for_spec(spec, dtype, device, shape)
            inputs.append(tensor)
        return tuple(inputs)

    def _make_input_for_spec(
        self,
        spec: InputSpec,
        dtype: torch.dtype,
        device: str,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Generate input tensor based on InputSpec."""
        low, high = spec.default_real_range

        if dtype.is_complex:
            # Generate complex tensor
            real_dtype = (
                torch.float32 if dtype == torch.complex64 else torch.float64
            )
            real = (
                torch.rand(shape, dtype=real_dtype, device=device)
                * (high - low)
                + low
            )
            imag_low, imag_high = spec.default_imag_range
            imag = (
                torch.rand(shape, dtype=real_dtype, device=device)
                * (imag_high - imag_low)
                + imag_low
            )
            tensor = torch.complex(real, imag)
        else:
            tensor = (
                torch.rand(shape, dtype=dtype, device=device) * (high - low)
                + low
            )

        # Filter out excluded values
        for excluded in spec.excluded_values:
            mask = (
                torch.abs(
                    tensor.real if dtype.is_complex else tensor - excluded
                )
                < 0.1
            )
            tensor = torch.where(mask, tensor + 0.2, tensor)

        return tensor


class UnaryOpTestCase(OpTestCase):
    """Test case for unary operators like gamma(z)."""

    pass


class BinaryOpTestCase(OpTestCase):
    """Test case for binary operators like chebyshev_polynomial_t(v, z)."""

    pass
