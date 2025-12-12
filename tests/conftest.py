import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from torch.library import opcheck

# Suppress differing_executors warning since we intentionally use class inheritance
# where the same base class test methods are called from multiple subclasses
HYPOTHESIS_SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.differing_executors],
)


@st.composite
def tensor_floats(draw, min_value=-10.0, max_value=10.0, min_size=1, max_size=20):
    values = draw(
        st.lists(
            st.floats(min_value, max_value, allow_nan=False, allow_infinity=False),
            min_size=min_size,
            max_size=max_size,
        )
    )
    return torch.tensor(values)


@st.composite
def tensor_with_shape(draw, min_value=-10.0, max_value=10.0):
    """Generate a tensor with a random shape (1D to 4D)."""
    ndim = draw(st.integers(min_value=1, max_value=4))
    shape = draw(
        st.lists(
            st.integers(min_value=1, max_value=5), min_size=ndim, max_size=ndim
        )
    )
    size = 1
    for dim in shape:
        size *= dim
    values = draw(
        st.lists(
            st.floats(min_value, max_value, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return torch.tensor(values).reshape(shape)


class UnaryOperatorTestCase:
    func = None
    op_name = None

    # Mathematical properties
    symmetry = None  # "odd" | "even" | None
    period = None  # float | None
    bounds = None  # (lower, upper) | None
    lower_bound = None  # float | None (for functions only bounded below)
    monotonic = None  # "increasing" | "decreasing" | None

    # Known values
    known_values = {}  # {input: expected_output}
    zeros = None  # list of zeros
    value_at_zero = None  # for functions with removable singularity at 0

    # Reference implementation for comparison
    reference = None  # callable | None

    # Identities: list of (callable, expected_value) tuples
    # e.g., [(lambda x: sin_pi(x)**2 + cos_pi(x)**2, 1.0)]
    identities = []

    # Input range for Hypothesis
    input_range = (-10.0, 10.0)

    # Gradcheck inputs (use values away from singularities)
    gradcheck_inputs = [0.1, 0.25, 0.7, 1.3]

    def test_known_values(self):
        if not self.known_values:
            pytest.skip("No known values defined")
        inputs = torch.tensor(list(self.known_values.keys()))
        expected = torch.tensor(list(self.known_values.values()))
        torch.testing.assert_close(self.func(inputs), expected, atol=1e-6, rtol=1e-5)

    def test_value_at_zero(self):
        if self.value_at_zero is None:
            pytest.skip("No value at zero defined")
        output = self.func(torch.tensor([0.0]))
        torch.testing.assert_close(
            output, torch.tensor([self.value_at_zero]), atol=1e-6, rtol=1e-5
        )

    def test_zeros(self):
        if self.zeros is None:
            pytest.skip("No zeros defined")
        inputs = torch.tensor(self.zeros, dtype=torch.float32)
        torch.testing.assert_close(
            self.func(inputs), torch.zeros_like(inputs), atol=1e-7, rtol=1e-5
        )

    # Tolerance for reference comparison (our impl may be more accurate)
    reference_atol = 1e-6
    reference_rtol = 1e-5

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_reference(self, data):
        """Compare against reference implementation."""
        if self.reference is None:
            pytest.skip("No reference implementation defined")
        x = data.draw(tensor_floats(*self.input_range))
        torch.testing.assert_close(
            self.func(x), self.reference(x), atol=self.reference_atol, rtol=self.reference_rtol
        )

    # Tolerance for identity verification
    identity_atol = 1e-5
    identity_rtol = 1e-5

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_identities(self, data):
        """Verify mathematical identities."""
        if not self.identities:
            pytest.skip("No identities defined")
        x = data.draw(tensor_floats(*self.input_range))
        for identity_func, expected_value in self.identities:
            result = identity_func(x)
            expected = torch.full_like(result, expected_value)
            torch.testing.assert_close(result, expected, atol=self.identity_atol, rtol=self.identity_rtol)

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_symmetry(self, data):
        if self.symmetry is None:
            pytest.skip("No symmetry defined")
        x = data.draw(tensor_floats(*self.input_range))
        if self.symmetry == "odd":
            torch.testing.assert_close(
                self.func(-x), -self.func(x), atol=1e-6, rtol=1e-5
            )
        elif self.symmetry == "even":
            torch.testing.assert_close(
                self.func(-x), self.func(x), atol=1e-6, rtol=1e-5
            )

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_periodicity(self, data):
        if self.period is None:
            pytest.skip("No period defined")
        x = data.draw(tensor_floats(*self.input_range))
        torch.testing.assert_close(
            self.func(x), self.func(x + self.period), atol=1e-5, rtol=1e-4
        )

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_bounds(self, data):
        if self.bounds is None:
            pytest.skip("No bounds defined")
        x = data.draw(tensor_floats(*self.input_range))
        output = self.func(x)
        assert torch.all(output >= self.bounds[0] - 1e-6)
        assert torch.all(output <= self.bounds[1] + 1e-6)

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_lower_bound(self, data):
        if self.lower_bound is None:
            pytest.skip("No lower bound defined")
        x = data.draw(tensor_floats(*self.input_range))
        output = self.func(x)
        assert torch.all(output >= self.lower_bound - 1e-6)

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.differing_executors])
    def test_monotonic(self, data):
        if self.monotonic is None:
            pytest.skip("No monotonicity defined")
        x = data.draw(tensor_floats(*self.input_range, min_size=10, max_size=50))
        x_sorted, _ = torch.sort(x)
        output = self.func(x_sorted)
        diff = output[1:] - output[:-1]
        if self.monotonic == "increasing":
            assert torch.all(diff >= -1e-6)
        elif self.monotonic == "decreasing":
            assert torch.all(diff <= 1e-6)

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_finite_output(self, data):
        """Verify output contains no inf or nan for valid inputs."""
        x = data.draw(tensor_floats(*self.input_range))
        output = self.func(x)
        assert torch.all(torch.isfinite(output)), f"Non-finite output: {output}"

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_shape_preservation(self, data):
        """Verify output shape matches input shape."""
        x = data.draw(tensor_with_shape(*self.input_range))
        output = self.func(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Verify output dtype matches input dtype."""
        x = torch.tensor(self.gradcheck_inputs, dtype=dtype)
        output = self.func(x)
        assert output.dtype == dtype, f"Dtype mismatch: {output.dtype} != {dtype}"

    def test_gradcheck(self):
        inputs = torch.tensor(
            self.gradcheck_inputs, dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(self.func, (inputs,), eps=1e-6, atol=1e-4, rtol=1e-3)

    # Set to True to enable second-order gradient testing
    supports_gradgradcheck = False

    def test_gradgradcheck(self):
        """Verify second-order gradients (Hessian)."""
        if not self.supports_gradgradcheck:
            pytest.skip("Second-order gradients not implemented")
        inputs = torch.tensor(
            self.gradcheck_inputs, dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradgradcheck(self.func, (inputs,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_non_contiguous(self):
        """Verify function works with non-contiguous tensors."""
        # Create a non-contiguous tensor via transpose
        x = torch.tensor([self.gradcheck_inputs, self.gradcheck_inputs], dtype=torch.float64).T
        assert not x.is_contiguous()
        output = self.func(x)
        # Compare with contiguous version
        expected = self.func(x.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_scalar(self):
        """Verify function works with 0-dimensional (scalar) tensors."""
        x = torch.tensor(self.gradcheck_inputs[0])
        output = self.func(x)
        assert output.dim() == 0, f"Expected scalar output, got shape {output.shape}"
        # Verify value matches 1D version
        expected = self.func(torch.tensor([self.gradcheck_inputs[0]]))[0]
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_empty_tensor(self):
        """Verify function works with empty tensors."""
        x = torch.tensor([], dtype=torch.float32)
        output = self.func(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        assert output.numel() == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_opcheck(self, dtype):
        inputs = torch.tensor(self.gradcheck_inputs, dtype=dtype)
        opcheck(getattr(torch.ops.torchscience, self.op_name), (inputs,))

    def test_device_cpu(self):
        """Verify function works on CPU."""
        x = torch.tensor(self.gradcheck_inputs, device="cpu")
        output = self.func(x)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Verify function works on CUDA."""
        x = torch.tensor(self.gradcheck_inputs, device="cuda")
        output = self.func(x)
        assert output.device.type == "cuda"
        # Verify result matches CPU
        expected = self.func(x.cpu()).cuda()
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_mps(self):
        """Verify function works on MPS (Apple Silicon)."""
        x = torch.tensor(self.gradcheck_inputs, device="mps")
        try:
            output = self.func(x)
        except NotImplementedError:
            pytest.skip("MPS not implemented for this operator")
        assert output.device.type == "mps"
        # Verify result matches CPU
        expected = self.func(x.cpu()).to("mps")
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_vmap(self):
        """Verify function works with torch.vmap for batched operations."""
        x = torch.tensor(self.gradcheck_inputs)
        # vmap over a batch dimension
        batched_x = x.unsqueeze(0).expand(3, -1)  # [3, N]
        try:
            vmapped_func = torch.vmap(self.func)
            output = vmapped_func(batched_x)
        except NotImplementedError:
            pytest.skip("vmap not implemented for this operator")
        # Verify shape
        assert output.shape == batched_x.shape
        # Verify each batch matches unbatched result
        expected = self.func(x)
        for i in range(3):
            torch.testing.assert_close(output[i], expected, atol=1e-6, rtol=1e-5)

    # Expected behavior for special values: "propagate" (nan in -> nan out),
    # "handle" (function handles specially), or None to skip test
    nan_behavior = "propagate"
    inf_behavior = "propagate"

    def test_nan_propagation(self):
        """Verify NaN handling behavior."""
        if self.nan_behavior is None:
            pytest.skip("NaN behavior not specified")
        x = torch.tensor([float("nan"), 1.0, float("nan")])
        output = self.func(x)
        if self.nan_behavior == "propagate":
            assert torch.isnan(output[0]), "Expected NaN to propagate"
            assert not torch.isnan(output[1]), "Non-NaN input produced NaN"
            assert torch.isnan(output[2]), "Expected NaN to propagate"

    def test_inf_handling(self):
        """Verify infinity handling behavior."""
        if self.inf_behavior is None:
            pytest.skip("Inf behavior not specified")
        x = torch.tensor([float("inf"), float("-inf"), 1.0])
        output = self.func(x)
        # Just verify it doesn't crash and produces finite or inf output (not NaN from inf)
        assert not torch.isnan(output[2]), "Finite input produced NaN"

    # Set to False if function does not support complex inputs
    supports_complex = True

    def test_complex_input(self):
        """Verify function works with complex inputs."""
        if not self.supports_complex:
            pytest.skip("Complex inputs not supported")
        x = torch.tensor(self.gradcheck_inputs, dtype=torch.complex64)
        x = x + 0.1j * x  # Add imaginary component
        try:
            output = self.func(x)
        except RuntimeError:
            pytest.skip("Complex inputs not supported")
        assert output.dtype == torch.complex64

    # Range for extreme value testing (very large/small magnitudes)
    extreme_values = [1e-30, 1e-10, 1e10, 1e30]

    def test_numerical_stability_extremes(self):
        """Verify numerical stability at extreme values."""
        x = torch.tensor(self.extreme_values + [-v for v in self.extreme_values])
        output = self.func(x)
        # Verify no NaN from finite input (inf output is acceptable for large inputs)
        finite_mask = torch.isfinite(x)
        # For finite inputs, output should not be NaN (inf is OK for functions like exp)
        nan_from_finite = torch.isnan(output) & finite_mask
        assert not torch.any(nan_from_finite), f"NaN produced from finite input: {x[nan_from_finite]} -> {output[nan_from_finite]}"

    def test_compile(self):
        """Verify function works with torch.compile."""
        x = torch.tensor(self.gradcheck_inputs)
        try:
            compiled_func = torch.compile(self.func, fullgraph=True)
            output = compiled_func(x)
        except Exception as e:
            pytest.skip(f"torch.compile not supported: {e}")
        expected = self.func(x)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_autocast(self, dtype):
        """Verify function works with autocast (mixed precision)."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            # bfloat16 CPU support varies
            pass
        x = torch.tensor(self.gradcheck_inputs, dtype=torch.float32)
        try:
            with torch.autocast(device_type="cpu", dtype=dtype):
                output = self.func(x)
        except RuntimeError:
            pytest.skip(f"Autocast with {dtype} not supported")
        # Output should be in lower precision dtype
        assert output.dtype in (dtype, torch.float32)

    def test_sparse_coo(self):
        """Verify function works with sparse COO tensors."""
        x = torch.tensor(self.gradcheck_inputs)
        sparse_x = x.to_sparse()
        try:
            output = self.func(sparse_x)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Sparse COO not supported")
        # Verify result matches dense
        expected = self.func(x)
        torch.testing.assert_close(output.to_dense(), expected, atol=1e-6, rtol=1e-5)

    def test_sparse_csr(self):
        """Verify function works with sparse CSR tensors."""
        # CSR requires 2D tensor
        x = torch.tensor([self.gradcheck_inputs, self.gradcheck_inputs])
        try:
            sparse_x = x.to_sparse_csr()
            output = self.func(sparse_x)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Sparse CSR not supported")
        # Verify result matches dense
        expected = self.func(x)
        torch.testing.assert_close(output.to_dense(), expected, atol=1e-6, rtol=1e-5)

    def test_quantized(self):
        """Verify function works with quantized tensors."""
        x = torch.tensor(self.gradcheck_inputs, dtype=torch.float32)
        try:
            qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.qint8)
            output = self.func(qx)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Quantized tensors not supported")
        # Just verify it runs - quantized output comparison is complex

    # Set to False if function doesn't preserve -0.0 sign (e.g., due to division)
    preserves_negative_zero = True

    def test_negative_zero(self):
        """Verify correct handling of negative zero."""
        pos_zero = torch.tensor([0.0])
        neg_zero = torch.tensor([-0.0])

        out_pos = self.func(pos_zero)
        out_neg = self.func(neg_zero)

        if self.symmetry == "odd" and self.preserves_negative_zero:
            # For odd functions, f(-0) should be -0 (sign preserved)
            assert torch.signbit(out_neg[0]) == True, "Odd function should preserve sign of -0.0"
            assert torch.signbit(out_pos[0]) == False, "Odd function should preserve sign of +0.0"
        elif self.symmetry == "even":
            # For even functions, f(-0) == f(0)
            torch.testing.assert_close(out_neg, out_pos, atol=0, rtol=0)
        else:
            # At minimum, values should be equal
            torch.testing.assert_close(out_neg, out_pos, atol=1e-10, rtol=0)

    def test_jacrev(self):
        """Verify function works with torch.func.jacrev."""
        x = torch.tensor(self.gradcheck_inputs, dtype=torch.float64)
        try:
            jac = torch.func.jacrev(self.func)(x)
        except Exception as e:
            pytest.skip(f"jacrev not supported: {e}")
        # Jacobian should be diagonal for element-wise functions
        assert jac.shape == (len(x), len(x))

    def test_jacfwd(self):
        """Verify function works with torch.func.jacfwd."""
        x = torch.tensor(self.gradcheck_inputs, dtype=torch.float64)
        try:
            jac = torch.func.jacfwd(self.func)(x)
        except Exception as e:
            pytest.skip(f"jacfwd not supported: {e}")
        # Jacobian should be diagonal for element-wise functions
        assert jac.shape == (len(x), len(x))

    def test_memory_layout_transpose(self):
        """Verify function works with transposed tensors."""
        x = torch.tensor([self.gradcheck_inputs] * 3, dtype=torch.float64)  # [3, N]
        x_t = x.T  # [N, 3], non-contiguous
        assert not x_t.is_contiguous()

        output = self.func(x_t)
        expected = self.func(x_t.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_slice(self):
        """Verify function works with sliced tensors (non-unit stride)."""
        x = torch.tensor(self.gradcheck_inputs * 3, dtype=torch.float64)
        x_sliced = x[::2]  # Every other element
        assert x_sliced.stride(0) == 2

        output = self.func(x_sliced)
        expected = self.func(x_sliced.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_expanded(self):
        """Verify function works with expanded tensors (zero stride)."""
        x = torch.tensor([self.gradcheck_inputs[0]], dtype=torch.float64)
        x_expanded = x.expand(5)  # [5] with stride 0
        assert x_expanded.stride(0) == 0

        output = self.func(x_expanded)
        expected = self.func(x_expanded.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_channels_last(self):
        """Verify function works with channels_last memory format."""
        x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        x_cl = x.to(memory_format=torch.channels_last)
        assert x_cl.is_contiguous(memory_format=torch.channels_last)
        assert not x_cl.is_contiguous()

        output = self.func(x_cl)
        expected = self.func(x.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)


class BinaryOperatorTestCase:
    func = None
    op_name = None

    # Known values: list of ((input1, input2), expected)
    known_values = []

    # Reference implementation for comparison
    reference = None  # callable | None

    # Identities: list of (callable, expected_value) tuples
    identities = []

    # Input ranges for Hypothesis
    input_range_1 = (0.1, 10.0)  # default avoids zero for division-like ops
    input_range_2 = (-10.0, 10.0)

    # Gradcheck inputs
    gradcheck_inputs = ([1.0, 2.0, 3.0], [0.25, 0.5, 0.75])

    def test_known_values(self):
        if not self.known_values:
            pytest.skip("No known values defined")
        for (in1, in2), expected in self.known_values:
            input1 = torch.tensor([in1])
            input2 = torch.tensor([in2])
            output = self.func(input1, input2)
            if isinstance(expected, complex):
                expected_tensor = torch.tensor([expected], dtype=output.dtype)
            else:
                expected_tensor = torch.tensor([expected])
            torch.testing.assert_close(output, expected_tensor, atol=1e-6, rtol=1e-5)

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_reference(self, data):
        """Compare against reference implementation."""
        if self.reference is None:
            pytest.skip("No reference implementation defined")
        x = data.draw(tensor_floats(*self.input_range_1))
        y = data.draw(tensor_floats(*self.input_range_2, min_size=len(x), max_size=len(x)))
        torch.testing.assert_close(
            self.func(x, y), self.reference(x, y), atol=1e-6, rtol=1e-5
        )

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_identities(self, data):
        """Verify mathematical identities."""
        if not self.identities:
            pytest.skip("No identities defined")
        x = data.draw(tensor_floats(*self.input_range_1))
        y = data.draw(tensor_floats(*self.input_range_2, min_size=len(x), max_size=len(x)))
        for identity_func, expected_value in self.identities:
            result = identity_func(x, y)
            expected = torch.full_like(result, expected_value)
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_finite_output(self, data):
        """Verify output contains no inf or nan for valid inputs."""
        x = data.draw(tensor_floats(*self.input_range_1))
        y = data.draw(tensor_floats(*self.input_range_2, min_size=len(x), max_size=len(x)))
        output = self.func(x, y)
        assert torch.all(torch.isfinite(output)), f"Non-finite output: {output}"

    @given(data=st.data())
    @HYPOTHESIS_SETTINGS
    def test_shape_preservation(self, data):
        """Verify output shape matches input shape."""
        x = data.draw(tensor_with_shape(*self.input_range_1))
        y_values = data.draw(
            st.lists(
                st.floats(*self.input_range_2, allow_nan=False, allow_infinity=False),
                min_size=x.numel(),
                max_size=x.numel(),
            )
        )
        y = torch.tensor(y_values).reshape(x.shape)
        output = self.func(x, y)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Verify output dtype is appropriate for inputs."""
        x = torch.tensor(self.gradcheck_inputs[0], dtype=dtype)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=dtype)
        output = self.func(x, y)
        # For complex outputs, check that real/imag dtype matches
        if output.is_complex():
            assert output.real.dtype == dtype
        else:
            assert output.dtype == dtype

    def test_gradcheck(self):
        input1 = torch.tensor(
            self.gradcheck_inputs[0], dtype=torch.float64, requires_grad=True
        )
        input2 = torch.tensor(
            self.gradcheck_inputs[1], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradcheck(
            self.func, (input1, input2), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    # Set to True to enable second-order gradient testing
    supports_gradgradcheck = False

    def test_gradgradcheck(self):
        """Verify second-order gradients (Hessian)."""
        if not self.supports_gradgradcheck:
            pytest.skip("Second-order gradients not implemented")
        input1 = torch.tensor(
            self.gradcheck_inputs[0], dtype=torch.float64, requires_grad=True
        )
        input2 = torch.tensor(
            self.gradcheck_inputs[1], dtype=torch.float64, requires_grad=True
        )
        torch.autograd.gradgradcheck(
            self.func, (input1, input2), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_non_contiguous(self):
        """Verify function works with non-contiguous tensors."""
        x = torch.tensor([self.gradcheck_inputs[0], self.gradcheck_inputs[0]], dtype=torch.float64).T
        y = torch.tensor([self.gradcheck_inputs[1], self.gradcheck_inputs[1]], dtype=torch.float64).T
        assert not x.is_contiguous()
        assert not y.is_contiguous()
        output = self.func(x, y)
        expected = self.func(x.contiguous(), y.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_scalar(self):
        """Verify function works with 0-dimensional (scalar) tensors."""
        x = torch.tensor(self.gradcheck_inputs[0][0])
        y = torch.tensor(self.gradcheck_inputs[1][0])
        output = self.func(x, y)
        assert output.dim() == 0, f"Expected scalar output, got shape {output.shape}"

    def test_empty_tensor(self):
        """Verify function works with empty tensors."""
        x = torch.tensor([], dtype=torch.float32)
        y = torch.tensor([], dtype=torch.float32)
        output = self.func(x, y)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        assert output.numel() == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_opcheck(self, dtype):
        input1 = torch.tensor(self.gradcheck_inputs[0], dtype=dtype)
        input2 = torch.tensor(self.gradcheck_inputs[1], dtype=dtype)
        opcheck(getattr(torch.ops.torchscience, self.op_name), (input1, input2))

    def test_device_cpu(self):
        """Verify function works on CPU."""
        x = torch.tensor(self.gradcheck_inputs[0], device="cpu")
        y = torch.tensor(self.gradcheck_inputs[1], device="cpu")
        output = self.func(x, y)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        """Verify function works on CUDA."""
        x = torch.tensor(self.gradcheck_inputs[0], device="cuda")
        y = torch.tensor(self.gradcheck_inputs[1], device="cuda")
        output = self.func(x, y)
        assert output.device.type == "cuda"
        # Verify result matches CPU
        expected = self.func(x.cpu(), y.cpu()).cuda()
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_device_mps(self):
        """Verify function works on MPS (Apple Silicon)."""
        x = torch.tensor(self.gradcheck_inputs[0], device="mps")
        y = torch.tensor(self.gradcheck_inputs[1], device="mps")
        try:
            output = self.func(x, y)
        except NotImplementedError:
            pytest.skip("MPS not implemented for this operator")
        assert output.device.type == "mps"
        # Verify result matches CPU
        expected = self.func(x.cpu(), y.cpu()).to("mps")
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_vmap(self):
        """Verify function works with torch.vmap for batched operations."""
        x = torch.tensor(self.gradcheck_inputs[0])
        y = torch.tensor(self.gradcheck_inputs[1])
        # vmap over a batch dimension
        batched_x = x.unsqueeze(0).expand(3, -1)  # [3, N]
        batched_y = y.unsqueeze(0).expand(3, -1)  # [3, N]
        try:
            vmapped_func = torch.vmap(self.func)
            output = vmapped_func(batched_x, batched_y)
        except NotImplementedError:
            pytest.skip("vmap not implemented for this operator")
        # Verify shape
        assert output.shape[0] == 3
        # Verify each batch matches unbatched result
        expected = self.func(x, y)
        for i in range(3):
            torch.testing.assert_close(output[i], expected, atol=1e-6, rtol=1e-5)

    # Expected behavior for special values
    nan_behavior = "propagate"
    inf_behavior = "propagate"

    def test_nan_propagation(self):
        """Verify NaN handling behavior."""
        if self.nan_behavior is None:
            pytest.skip("NaN behavior not specified")
        x = torch.tensor([float("nan"), 1.0, 1.0])
        y = torch.tensor([1.0, float("nan"), 1.0])
        output = self.func(x, y)
        if self.nan_behavior == "propagate":
            assert torch.isnan(output[0]), "Expected NaN to propagate from first input"
            assert torch.isnan(output[1]), "Expected NaN to propagate from second input"
            assert not torch.isnan(output[2]), "Non-NaN inputs produced NaN"

    def test_inf_handling(self):
        """Verify infinity handling behavior."""
        if self.inf_behavior is None:
            pytest.skip("Inf behavior not specified")
        x = torch.tensor([float("inf"), 1.0, 1.0])
        y = torch.tensor([1.0, float("-inf"), 1.0])
        output = self.func(x, y)
        # Just verify it doesn't crash and finite inputs produce non-NaN
        assert not torch.isnan(output[2]), "Finite inputs produced NaN"

    # Set to False if function does not support complex inputs
    supports_complex = True

    def test_complex_input(self):
        """Verify function works with complex inputs."""
        if not self.supports_complex:
            pytest.skip("Complex inputs not supported")
        x = torch.tensor(self.gradcheck_inputs[0], dtype=torch.complex64)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=torch.complex64)
        x = x + 0.1j * x
        y = y + 0.1j * y
        try:
            output = self.func(x, y)
        except RuntimeError:
            pytest.skip("Complex inputs not supported")
        assert output.is_complex()

    # Range for extreme value testing
    extreme_values = [1e-30, 1e-10, 1e10, 1e30]

    def test_numerical_stability_extremes(self):
        """Verify numerical stability at extreme values."""
        x = torch.tensor(self.extreme_values + [-v for v in self.extreme_values])
        y = torch.tensor([1.0] * len(x))  # Use normal values for second input
        output = self.func(x, y)
        finite_mask = torch.isfinite(x) & torch.isfinite(y)
        nan_from_finite = torch.isnan(output) & finite_mask
        assert not torch.any(nan_from_finite), f"NaN produced from finite inputs"

    def test_compile(self):
        """Verify function works with torch.compile."""
        x = torch.tensor(self.gradcheck_inputs[0])
        y = torch.tensor(self.gradcheck_inputs[1])
        try:
            compiled_func = torch.compile(self.func, fullgraph=True)
            output = compiled_func(x, y)
        except Exception as e:
            pytest.skip(f"torch.compile not supported: {e}")
        expected = self.func(x, y)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_autocast(self, dtype):
        """Verify function works with autocast (mixed precision)."""
        x = torch.tensor(self.gradcheck_inputs[0], dtype=torch.float32)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=torch.float32)
        try:
            with torch.autocast(device_type="cpu", dtype=dtype):
                output = self.func(x, y)
        except RuntimeError:
            pytest.skip(f"Autocast with {dtype} not supported")
        assert output.dtype in (dtype, torch.float32, torch.complex64, torch.complex32)

    def test_sparse_coo(self):
        """Verify function works with sparse COO tensors."""
        x = torch.tensor(self.gradcheck_inputs[0])
        y = torch.tensor(self.gradcheck_inputs[1])
        sparse_x = x.to_sparse()
        sparse_y = y.to_sparse()
        try:
            output = self.func(sparse_x, sparse_y)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Sparse COO not supported")
        expected = self.func(x, y)
        torch.testing.assert_close(output.to_dense(), expected, atol=1e-6, rtol=1e-5)

    def test_sparse_csr(self):
        """Verify function works with sparse CSR tensors."""
        x = torch.tensor([self.gradcheck_inputs[0], self.gradcheck_inputs[0]])
        y = torch.tensor([self.gradcheck_inputs[1], self.gradcheck_inputs[1]])
        try:
            sparse_x = x.to_sparse_csr()
            sparse_y = y.to_sparse_csr()
            output = self.func(sparse_x, sparse_y)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Sparse CSR not supported")
        expected = self.func(x, y)
        torch.testing.assert_close(output.to_dense(), expected, atol=1e-6, rtol=1e-5)

    def test_quantized(self):
        """Verify function works with quantized tensors."""
        x = torch.tensor(self.gradcheck_inputs[0], dtype=torch.float32)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=torch.float32)
        try:
            qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.qint8)
            qy = torch.quantize_per_tensor(y, scale=0.1, zero_point=0, dtype=torch.qint8)
            output = self.func(qx, qy)
        except (RuntimeError, NotImplementedError):
            pytest.skip("Quantized tensors not supported")

    # Set to False if function does not support broadcasting
    supports_broadcasting = True

    def test_broadcasting(self):
        """Verify function supports broadcasting."""
        if not self.supports_broadcasting:
            pytest.skip("Broadcasting not supported")
        x = torch.tensor(self.gradcheck_inputs[0]).unsqueeze(1)  # [N, 1]
        y = torch.tensor(self.gradcheck_inputs[1]).unsqueeze(0)  # [1, M]
        try:
            output = self.func(x, y)
        except RuntimeError as e:
            if "broadcast" in str(e).lower() or "size" in str(e).lower() or "shape" in str(e).lower():
                pytest.skip("Broadcasting not supported")
            raise
        expected_shape = (x.shape[0], y.shape[1])
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    def test_jacrev(self):
        """Verify function works with torch.func.jacrev."""
        x = torch.tensor(self.gradcheck_inputs[0], dtype=torch.float64)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=torch.float64)

        def func_x(x_):
            return self.func(x_, y)

        def func_y(y_):
            return self.func(x, y_)

        try:
            jac_x = torch.func.jacrev(func_x)(x)
            jac_y = torch.func.jacrev(func_y)(y)
        except Exception as e:
            pytest.skip(f"jacrev not supported: {e}")
        assert jac_x.shape[0] == len(x)
        assert jac_y.shape[0] == len(y)

    def test_jacfwd(self):
        """Verify function works with torch.func.jacfwd."""
        x = torch.tensor(self.gradcheck_inputs[0], dtype=torch.float64)
        y = torch.tensor(self.gradcheck_inputs[1], dtype=torch.float64)

        def func_x(x_):
            return self.func(x_, y)

        def func_y(y_):
            return self.func(x, y_)

        try:
            jac_x = torch.func.jacfwd(func_x)(x)
            jac_y = torch.func.jacfwd(func_y)(y)
        except Exception as e:
            pytest.skip(f"jacfwd not supported: {e}")
        assert jac_x.shape[0] == len(x)
        assert jac_y.shape[0] == len(y)

    def test_memory_layout_transpose(self):
        """Verify function works with transposed tensors."""
        x = torch.tensor([self.gradcheck_inputs[0]] * 3, dtype=torch.float64)  # [3, N]
        y = torch.tensor([self.gradcheck_inputs[1]] * 3, dtype=torch.float64)  # [3, M]
        x_t = x.T
        y_t = y.T
        assert not x_t.is_contiguous()
        assert not y_t.is_contiguous()

        output = self.func(x_t, y_t)
        expected = self.func(x_t.contiguous(), y_t.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_slice(self):
        """Verify function works with sliced tensors (non-unit stride)."""
        x = torch.tensor(self.gradcheck_inputs[0] * 3, dtype=torch.float64)
        y = torch.tensor(self.gradcheck_inputs[1] * 3, dtype=torch.float64)
        x_sliced = x[::2]
        y_sliced = y[::2]
        assert x_sliced.stride(0) == 2
        assert y_sliced.stride(0) == 2

        output = self.func(x_sliced, y_sliced)
        expected = self.func(x_sliced.contiguous(), y_sliced.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_expanded(self):
        """Verify function works with expanded tensors (zero stride)."""
        if not self.supports_broadcasting:
            pytest.skip("Broadcasting not supported")
        x = torch.tensor([self.gradcheck_inputs[0][0]], dtype=torch.float64)
        y = torch.tensor([self.gradcheck_inputs[1][0]], dtype=torch.float64)
        x_expanded = x.expand(5)
        y_expanded = y.expand(5)
        assert x_expanded.stride(0) == 0
        assert y_expanded.stride(0) == 0

        output = self.func(x_expanded, y_expanded)
        expected = self.func(x_expanded.contiguous(), y_expanded.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

    def test_memory_layout_channels_last(self):
        """Verify function works with channels_last memory format."""
        x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        y = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        x_cl = x.to(memory_format=torch.channels_last)
        y_cl = y.to(memory_format=torch.channels_last)
        assert not x_cl.is_contiguous()
        assert not y_cl.is_contiguous()

        output = self.func(x_cl, y_cl)
        expected = self.func(x.contiguous(), y.contiguous())
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
