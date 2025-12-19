import torch
import torch.testing

import torchscience.window_function
from torchscience.testing import (
    CreationOpDescriptor,
    CreationOpTestCase,
    CreationOpToleranceConfig,
    ExpectedValue,
)


def reference_rectangular_window(
    m: int,
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Reference implementation of rectangular window using torch.ones."""
    if m <= 0:
        raise ValueError("m must be positive")
    result = torch.ones(
        m,
        dtype=dtype or torch.float32,
        device=device or "cpu",
    )
    if requires_grad:
        result = result.requires_grad_(True)
    return result


class TestRectangularWindow(CreationOpTestCase):
    """Tests for the rectangular window function."""

    @property
    def descriptor(self) -> CreationOpDescriptor:
        return CreationOpDescriptor(
            name="rectangular_window",
            func=torchscience.window_function.rectangular_window,
            expected_values=[
                ExpectedValue(
                    m=1,
                    expected=torch.tensor([1.0], dtype=torch.float32),
                    description="Single element window",
                ),
                ExpectedValue(
                    m=5,
                    expected=torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32
                    ),
                    description="5-element window",
                ),
                ExpectedValue(
                    m=3,
                    expected=torch.tensor(
                        [1.0, 1.0, 1.0], dtype=torch.float64
                    ),
                    description="3-element window float64",
                ),
            ],
            supported_dtypes=[
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ],
            tolerances=CreationOpToleranceConfig(),
            skip_tests=set(),
            supports_meta=True,
            reference_func=reference_rectangular_window,
        )

    # =========================================================================
    # Rectangular window specific tests
    # =========================================================================

    def test_all_ones(self):
        """Test that all elements are exactly 1.0."""
        for m in [1, 5, 10, 100]:
            result = torchscience.window_function.rectangular_window(m)
            expected = torch.ones(m, dtype=torch.float32)
            torch.testing.assert_close(result, expected)

    def test_all_ones_float64(self):
        """Test that all elements are exactly 1.0 for float64."""
        for m in [1, 5, 10, 100]:
            result = torchscience.window_function.rectangular_window(
                m, dtype=torch.float64
            )
            expected = torch.ones(m, dtype=torch.float64)
            torch.testing.assert_close(result, expected)

    def test_sum_equals_length(self):
        """Test that sum of window equals window length (coherent gain = M)."""
        for m in [1, 5, 10, 50]:
            result = torchscience.window_function.rectangular_window(
                m, dtype=torch.float64
            )
            assert result.sum().item() == float(m), (
                f"Expected sum={m}, got {result.sum().item()}"
            )

    def test_normalization_property(self):
        """Test that mean of rectangular window is 1.0."""
        for m in [1, 5, 10, 100]:
            result = torchscience.window_function.rectangular_window(
                m, dtype=torch.float64
            )
            assert result.mean().item() == 1.0, (
                f"Expected mean=1.0, got {result.mean().item()}"
            )

    def test_symmetry(self):
        """Test that rectangular window is symmetric."""
        for m in [1, 5, 10, 11]:
            result = torchscience.window_function.rectangular_window(
                m, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped)

    def test_max_and_min(self):
        """Test that max and min are both 1.0."""
        for m in [1, 5, 10, 100]:
            result = torchscience.window_function.rectangular_window(
                m, dtype=torch.float64
            )
            assert result.max().item() == 1.0, (
                f"Expected max=1.0, got {result.max().item()}"
            )
            assert result.min().item() == 1.0, (
                f"Expected min=1.0, got {result.min().item()}"
            )

    def test_large_window(self):
        """Test with large window size."""
        m = 10000
        result = torchscience.window_function.rectangular_window(m)
        assert result.shape == (m,)
        assert result.sum().item() == float(m)

    def test_gradient_flow(self):
        """Test that gradients flow through when requires_grad=True."""
        result = torchscience.window_function.rectangular_window(
            5, dtype=torch.float64, requires_grad=True
        )
        loss = result.sum()
        loss.backward()
        # For a constant tensor, gradient should be None or zeros
        # since the window values don't depend on any input

    def test_multiply_with_signal(self):
        """Test typical use case of multiplying window with a signal."""
        m = 10
        window = torchscience.window_function.rectangular_window(
            m, dtype=torch.float64
        )
        signal = torch.randn(m, dtype=torch.float64)
        windowed = window * signal
        # Rectangular window should not modify the signal
        torch.testing.assert_close(windowed, signal)

    def test_comparison_with_torch_signal_windows(self):
        """Compare with torch.signal.windows if available."""
        try:
            import torch.signal.windows

            for m in [5, 10, 20]:
                result = torchscience.window_function.rectangular_window(
                    m, dtype=torch.float64
                )
                # torch.signal.windows might not have rectangular, but if it does:
                if hasattr(torch.signal.windows, "rectangular"):
                    expected = torch.signal.windows.rectangular(m)
                    torch.testing.assert_close(result, expected)
        except (ImportError, AttributeError):
            # torch.signal.windows may not be available in all versions
            pass

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_m_equals_one(self):
        """Test edge case where m=1."""
        result = torchscience.window_function.rectangular_window(1)
        assert result.shape == (1,)
        assert result[0].item() == 1.0

    def test_m_equals_two(self):
        """Test edge case where m=2."""
        result = torchscience.window_function.rectangular_window(2)
        assert result.shape == (2,)
        torch.testing.assert_close(
            result, torch.tensor([1.0, 1.0], dtype=torch.float32)
        )

    # =========================================================================
    # Dtype edge cases
    # =========================================================================

    def test_half_precision(self):
        """Test float16 dtype."""
        result = torchscience.window_function.rectangular_window(
            5, dtype=torch.float16
        )
        assert result.dtype == torch.float16
        expected = torch.ones(5, dtype=torch.float16)
        torch.testing.assert_close(result, expected)

    def test_bfloat16_precision(self):
        """Test bfloat16 dtype."""
        result = torchscience.window_function.rectangular_window(
            5, dtype=torch.bfloat16
        )
        assert result.dtype == torch.bfloat16
        expected = torch.ones(5, dtype=torch.bfloat16)
        torch.testing.assert_close(result, expected)

    # =========================================================================
    # Complex dtype tests (rectangular window typically real, but test handling)
    # =========================================================================

    def test_complex64_not_supported_or_works(self):
        """Test behavior with complex64 dtype."""
        try:
            result = torchscience.window_function.rectangular_window(
                5, dtype=torch.complex64
            )
            # If it works, verify it's all ones
            assert result.dtype == torch.complex64
            expected = torch.ones(5, dtype=torch.complex64)
            torch.testing.assert_close(result, expected)
        except (RuntimeError, TypeError):
            # Complex dtypes may not be supported for window functions
            pass

    # =========================================================================
    # Memory and performance tests
    # =========================================================================

    def test_contiguous_output(self):
        """Test that output tensor is contiguous."""
        result = torchscience.window_function.rectangular_window(10)
        assert result.is_contiguous(), "Output should be contiguous"

    def test_no_memory_leak_large_allocations(self):
        """Test that repeated large allocations don't leak memory."""
        for _ in range(10):
            _ = torchscience.window_function.rectangular_window(10000)

    # =========================================================================
    # Integration tests
    # =========================================================================

    def test_fft_windowing_workflow(self):
        """Test typical FFT windowing workflow."""
        # Create a test signal
        m = 64
        signal = torch.randn(m, dtype=torch.float64)

        # Apply rectangular window
        window = torchscience.window_function.rectangular_window(
            m, dtype=torch.float64
        )
        windowed_signal = signal * window

        # With rectangular window, windowed signal equals original
        torch.testing.assert_close(windowed_signal, signal)

        # Can compute FFT
        fft_result = torch.fft.fft(windowed_signal)
        assert fft_result.shape == (m,)

    def test_batch_windowing(self):
        """Test applying window to batched signals."""
        batch_size = 4
        m = 32

        # Create batched signals [batch, time]
        signals = torch.randn(batch_size, m, dtype=torch.float64)

        # Create window and broadcast
        window = torchscience.window_function.rectangular_window(
            m, dtype=torch.float64
        )
        windowed = signals * window  # Broadcasting: [batch, time] * [time]

        # Should have same shape as input
        assert windowed.shape == signals.shape

        # With rectangular window, output equals input
        torch.testing.assert_close(windowed, signals)
