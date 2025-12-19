"""Base test class for creation operators.

Creation operators are factory functions that create new tensors based on
scalar parameters rather than operating element-wise on existing tensors.
Examples include window functions, factory functions, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Tuple

import torch
from torch import Tensor


@dataclass
class CreationOpToleranceConfig:
    """Tolerance configuration for creation operator numerical comparisons."""

    float16_rtol: float = 1e-3
    float16_atol: float = 1e-3
    bfloat16_rtol: float = 1e-2
    bfloat16_atol: float = 1e-2
    float32_rtol: float = 1e-5
    float32_atol: float = 1e-5
    float64_rtol: float = 1e-10
    float64_atol: float = 1e-10
    complex64_rtol: float = 1e-5
    complex64_atol: float = 1e-5
    complex128_rtol: float = 1e-10
    complex128_atol: float = 1e-10

    def get_tolerances(self, dtype: torch.dtype) -> Tuple[float, float]:
        """Return (rtol, atol) for the given dtype."""
        mapping = {
            torch.float16: (self.float16_rtol, self.float16_atol),
            torch.bfloat16: (self.bfloat16_rtol, self.bfloat16_atol),
            torch.float32: (self.float32_rtol, self.float32_atol),
            torch.float64: (self.float64_rtol, self.float64_atol),
            torch.complex64: (self.complex64_rtol, self.complex64_atol),
            torch.complex128: (self.complex128_rtol, self.complex128_atol),
        }
        return mapping.get(dtype, (1e-5, 1e-5))


@dataclass
class ExpectedValue:
    """A test case with expected values for a creation operator."""

    m: int
    expected: Tensor
    rtol: float = 1e-10
    atol: float = 1e-10
    description: str = ""


@dataclass
class CreationOpDescriptor:
    """Describes a creation operator for testing."""

    # Basic info
    name: str
    func: Callable[..., Tensor]

    # Expected values for specific inputs
    expected_values: list[ExpectedValue] = field(default_factory=list)

    # Supported dtypes
    supported_dtypes: list[torch.dtype] = field(
        default_factory=lambda: [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]
    )

    # Tolerance configurations
    tolerances: CreationOpToleranceConfig = field(
        default_factory=CreationOpToleranceConfig
    )

    # Skip configurations
    skip_tests: Set[str] = field(default_factory=set)

    # Meta tensor support
    supports_meta: bool = True

    # Reference implementation for comparison
    reference_func: Optional[Callable[..., Tensor]] = None


class CreationOpTestCase(ABC):
    """Base test case for creation operators (factory functions).

    Creation operators create new tensors based on scalar parameters.
    This base class provides common tests for:
    - Output shape correctness
    - Dtype support
    - Device support
    - Expected values
    - Edge cases (empty tensors)
    - Meta tensor support
    """

    @property
    @abstractmethod
    def descriptor(self) -> CreationOpDescriptor:
        """Return the creation operator descriptor."""
        ...

    # =========================================================================
    # Shape tests
    # =========================================================================

    def test_output_shape(self):
        """Test that output has correct shape."""
        for m in [1, 5, 10, 100]:
            result = self.descriptor.func(m)
            assert result.shape == (m,), (
                f"Expected shape ({m},), got {result.shape} for m={m}"
            )

    def test_error_for_zero_size(self):
        """Test that m=0 raises an error."""
        import pytest

        with pytest.raises(RuntimeError):
            self.descriptor.func(0)

    def test_error_for_negative_size(self):
        """Test that m<0 raises an error."""
        import pytest

        with pytest.raises(RuntimeError):
            self.descriptor.func(-1)
        with pytest.raises(RuntimeError):
            self.descriptor.func(-10)

    # =========================================================================
    # Dtype tests
    # =========================================================================

    def test_default_dtype(self):
        """Test that default dtype is float32."""
        result = self.descriptor.func(5)
        assert result.dtype == torch.float32, (
            f"Expected default dtype float32, got {result.dtype}"
        )

    def test_explicit_dtype(self):
        """Test that explicit dtype is respected."""
        for dtype in self.descriptor.supported_dtypes:
            if f"test_dtype_{dtype}" in self.descriptor.skip_tests:
                continue
            result = self.descriptor.func(5, dtype=dtype)
            assert result.dtype == dtype, (
                f"Expected dtype {dtype}, got {result.dtype}"
            )

    # =========================================================================
    # Device tests
    # =========================================================================

    def test_default_device(self):
        """Test that default device is CPU."""
        result = self.descriptor.func(5)
        assert result.device.type == "cpu", (
            f"Expected default device cpu, got {result.device}"
        )

    def test_cpu_device(self):
        """Test explicit CPU device."""
        result = self.descriptor.func(5, device="cpu")
        assert result.device.type == "cpu", (
            f"Expected device cpu, got {result.device}"
        )

    def test_cuda_device(self):
        """Test CUDA device if available."""
        if not torch.cuda.is_available():
            return
        if "test_cuda_device" in self.descriptor.skip_tests:
            return
        result = self.descriptor.func(5, device="cuda")
        assert result.device.type == "cuda", (
            f"Expected device cuda, got {result.device}"
        )

    # =========================================================================
    # requires_grad tests
    # =========================================================================

    def test_default_requires_grad(self):
        """Test that default requires_grad is False."""
        result = self.descriptor.func(5)
        assert not result.requires_grad, (
            "Expected requires_grad=False by default"
        )

    def test_explicit_requires_grad_true(self):
        """Test that requires_grad=True is respected."""
        result = self.descriptor.func(5, requires_grad=True)
        assert result.requires_grad, "Expected requires_grad=True"

    def test_explicit_requires_grad_false(self):
        """Test that requires_grad=False is respected."""
        result = self.descriptor.func(5, requires_grad=False)
        assert not result.requires_grad, "Expected requires_grad=False"

    # =========================================================================
    # Expected value tests
    # =========================================================================

    def test_expected_values(self):
        """Test against expected values specified in descriptor."""
        for ev in self.descriptor.expected_values:
            result = self.descriptor.func(ev.m, dtype=ev.expected.dtype)
            torch.testing.assert_close(
                result,
                ev.expected,
                rtol=ev.rtol,
                atol=ev.atol,
                msg=f"Value mismatch for m={ev.m}: {ev.description}",
            )

    # =========================================================================
    # Reference implementation tests
    # =========================================================================

    def test_reference_implementation(self):
        """Test against reference implementation if provided."""
        if self.descriptor.reference_func is None:
            return
        for m in [1, 5, 10, 20, 100]:
            for dtype in [torch.float32, torch.float64]:
                result = self.descriptor.func(m, dtype=dtype)
                expected = self.descriptor.reference_func(m, dtype=dtype)
                rtol, atol = self.descriptor.tolerances.get_tolerances(dtype)
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Reference mismatch for m={m}, dtype={dtype}",
                )

    # =========================================================================
    # Meta tensor tests
    # =========================================================================

    def test_meta_tensor_shape(self):
        """Test that meta tensors have correct shape."""
        if not self.descriptor.supports_meta:
            return
        if "test_meta_tensor" in self.descriptor.skip_tests:
            return
        for m in [1, 5, 10]:
            result = self.descriptor.func(m, device="meta")
            assert result.shape == (m,), (
                f"Meta tensor shape mismatch: expected ({m},), "
                f"got {result.shape}"
            )
            assert result.device.type == "meta", (
                f"Expected meta device, got {result.device}"
            )

    # =========================================================================
    # torch.compile tests
    # =========================================================================

    def test_torch_compile(self):
        """Test that the function works with torch.compile."""
        if "test_torch_compile" in self.descriptor.skip_tests:
            return
        compiled_func = torch.compile(self.descriptor.func)
        result = compiled_func(5)
        expected = self.descriptor.func(5)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Dtype and device combination tests
    # =========================================================================

    def test_dtype_device_combinations(self):
        """Test various dtype and device combinations."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for dtype in self.descriptor.supported_dtypes:
            for device in devices:
                test_name = f"test_dtype_device_{dtype}_{device}"
                if test_name in self.descriptor.skip_tests:
                    continue
                result = self.descriptor.func(5, dtype=dtype, device=device)
                assert result.dtype == dtype, (
                    f"Expected dtype {dtype}, got {result.dtype}"
                )
                assert result.device.type == device, (
                    f"Expected device {device}, got {result.device}"
                )
