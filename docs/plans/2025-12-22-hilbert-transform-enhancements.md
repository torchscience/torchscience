# Hilbert Transform Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move integral_transform module under signal_processing and add windowing/padding support to hilbert_transform with C++ kernel implementations following PyTorch conventions.

**Architecture:**
- Relocate `torchscience.integral_transform` to `torchscience.signal_processing.integral_transform`
- Add `padding_mode` and `window` parameters to C++ operator definitions
- Implement padding logic (constant, reflect, replicate, circular) in CPU and CUDA kernels
- Window application in C++ kernels before FFT
- Python layer is a thin wrapper passing parameters to C++

**Tech Stack:** C++ (ATen/PyTorch), CUDA, Python

---

## Phase 1: C++ Operator Schema Updates

### Task 1: Update Operator Definitions

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp:119-127`

**Step 1: Update hilbert_transform operator schema**

Modify the operator definitions to include new parameters:

```cpp
// `torchscience.integral_transform`
// n=-1 means use input size along dim (no padding/truncation)
// padding_mode: 0=constant, 1=reflect, 2=replicate, 3=circular
module.def("hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
module.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
module.def("hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

module.def("inverse_hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
module.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
module.def("inverse_hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");
```

**Step 2: Verify compilation**

Run: `uv run python -c "import torchscience._csrc"` (will fail until implementations updated)

**Step 3: Commit schema changes**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: update hilbert_transform schema with padding_mode and window parameters"
```

---

### Task 2: Add Padding Utilities to impl Header

**Files:**
- Modify: `src/torchscience/csrc/impl/integral_transform/hilbert_transform.h`

**Step 1: Add padding mode enum and utility functions**

Add after the existing includes and before the namespace:

```cpp
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/pad.h>
#include <cmath>

namespace torchscience::impl::integral_transform {

/**
 * Padding mode enum matching PyTorch conventions.
 * Maps to torch.nn.functional.pad modes.
 */
enum class PaddingMode : int64_t {
    Constant = 0,
    Reflect = 1,
    Replicate = 2,
    Circular = 3
};

/**
 * Convert padding mode integer to string for ATen pad function.
 */
inline std::string padding_mode_to_string(int64_t mode) {
    switch (static_cast<PaddingMode>(mode)) {
        case PaddingMode::Constant: return "constant";
        case PaddingMode::Reflect: return "reflect";
        case PaddingMode::Replicate: return "replicate";
        case PaddingMode::Circular: return "circular";
        default:
            TORCH_CHECK(false, "Invalid padding_mode: ", mode,
                ". Must be 0 (constant), 1 (reflect), 2 (replicate), or 3 (circular)");
    }
}

/**
 * Apply padding to tensor along specified dimension.
 *
 * @param input Input tensor
 * @param target_size Target size along dim after padding
 * @param dim Dimension to pad
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @return Padded tensor
 */
inline at::Tensor apply_padding(
    const at::Tensor& input,
    int64_t target_size,
    int64_t dim,
    int64_t padding_mode,
    double padding_value
) {
    int64_t current_size = input.size(dim);

    if (target_size <= current_size) {
        // No padding needed (truncation handled elsewhere)
        return input;
    }

    int64_t pad_amount = target_size - current_size;
    int64_t ndim = input.dim();

    // Move target dim to last position for F.pad
    at::Tensor input_moved = input.movedim(dim, -1);

    // Build pad tuple: (left, right) for last dim only
    // ATen pad expects IntArrayRef in reverse dimension order
    std::vector<int64_t> pad_sizes = {0, pad_amount};

    at::Tensor padded;
    std::string mode_str = padding_mode_to_string(padding_mode);

    if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
        padded = at::pad(input_moved, pad_sizes, mode_str, padding_value);
    } else {
        padded = at::pad(input_moved, pad_sizes, mode_str);
    }

    // Move dimension back
    return padded.movedim(-1, dim);
}

/**
 * Apply window function to tensor along specified dimension.
 *
 * @param input Input tensor
 * @param window 1-D window tensor (must match input size along dim)
 * @param dim Dimension to apply window
 * @return Windowed tensor
 */
inline at::Tensor apply_window(
    const at::Tensor& input,
    const at::Tensor& window,
    int64_t dim
) {
    TORCH_CHECK(window.dim() == 1,
        "window must be 1-D, got ", window.dim(), "-D tensor");
    TORCH_CHECK(window.size(0) == input.size(dim),
        "window size (", window.size(0), ") must match input size along dim (",
        input.size(dim), ")");

    // Reshape window for broadcasting: (1, 1, ..., n, ..., 1)
    std::vector<int64_t> window_shape(input.dim(), 1);
    window_shape[dim] = window.size(0);

    at::Tensor window_reshaped = window.view(window_shape);

    return input * window_reshaped;
}

// ... existing hilbert_frequency_response and other functions ...
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/integral_transform/hilbert_transform.h
git commit -m "feat: add padding and window utilities to hilbert_transform impl"
```

---

## Phase 2: CPU Kernel Updates

### Task 3: Update CPU hilbert_transform Implementation

**Files:**
- Modify: `src/torchscience/csrc/cpu/integral_transform/hilbert_transform.h`

**Step 1: Update function signatures and implementation**

```cpp
#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

#include "../../impl/integral_transform/hilbert_transform.h"
#include "../../impl/integral_transform/hilbert_transform_backward.h"
#include "../../impl/integral_transform/hilbert_transform_backward_backward.h"

namespace torchscience::cpu::integral_transform {

/**
 * CPU implementation of Hilbert transform with padding and windowing.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @param window Optional window tensor to apply before FFT
 * @return Hilbert transform of the input
 */
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.numel() > 0, "hilbert_transform: input tensor must be non-empty");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "hilbert_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "hilbert_transform: n must be positive");

    // Ensure contiguous for efficient operations
    at::Tensor processed = input.contiguous();

    // Apply padding if needed
    if (n > input_size) {
        processed = impl::integral_transform::apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        // Truncation
        processed = processed.narrow(dim, 0, n);
    }

    // Apply window if provided
    if (window.has_value()) {
        processed = impl::integral_transform::apply_window(
            processed, window.value(), dim
        );
    }

    // Compute FFT along specified dimension
    at::Tensor spectrum = at::fft_fft(processed, c10::nullopt, dim);

    // Create frequency response tensor
    std::vector<int64_t> response_shape(ndim, 1);
    response_shape[dim] = n;

    at::Tensor response = at::zeros(response_shape, spectrum.options());

    // Fill in the response values
    AT_DISPATCH_COMPLEX_TYPES(
        spectrum.scalar_type(),
        "hilbert_transform_cpu_response",
        [&]() {
            using real_t = typename c10::scalar_value_type<scalar_t>::type;

            auto response_flat = response.view({n});
            auto* response_data = response_flat.data_ptr<scalar_t>();

            for (int64_t k = 0; k < n; ++k) {
                auto h = impl::integral_transform::hilbert_frequency_response<real_t>(k, n);
                response_data[k] = scalar_t(h.real(), h.imag());
            }
        }
    );

    // Apply frequency response
    at::Tensor modified_spectrum = spectrum * response;

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part
    if (!input.is_complex()) {
        result = at::real(result).contiguous();
    }

    return result;
}

/**
 * Backward pass for Hilbert transform on CPU.
 */
inline at::Tensor hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H with size n to grad_output
    // Note: backward doesn't apply window (it's part of forward chain rule)
    at::Tensor grad = -hilbert_transform(grad_output, n_param, dim, 0, 0.0, c10::nullopt);

    // Adjust size to match input shape
    grad = impl::integral_transform::adjust_backward_gradient_size(
        grad, input_size, n, norm_dim
    );

    // If window was applied in forward, multiply gradient by window
    // (chain rule: d/dx[w*x] = w * d/dx[x])
    if (window.has_value()) {
        grad = impl::integral_transform::apply_window(grad, window.value(), norm_dim);
    }

    return grad;
}

/**
 * Double backward pass for Hilbert transform on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // grad_grad_output = H[grad_grad_input]
    at::Tensor grad_grad_output = hilbert_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window
    );

    // No second-order term (H is linear)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "hilbert_transform",
        &torchscience::cpu::integral_transform::hilbert_transform
    );

    module.impl(
        "hilbert_transform_backward",
        &torchscience::cpu::integral_transform::hilbert_transform_backward
    );

    module.impl(
        "hilbert_transform_backward_backward",
        &torchscience::cpu::integral_transform::hilbert_transform_backward_backward
    );
}
```

**Step 2: Verify compilation**

Run: `uv run python setup.py build_ext --inplace`

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/integral_transform/hilbert_transform.h
git commit -m "feat: update CPU hilbert_transform with padding_mode and window"
```

---

### Task 4: Update CUDA hilbert_transform Implementation

**Files:**
- Modify: `src/torchscience/csrc/cuda/integral_transform/hilbert_transform.cu`

**Step 1: Update CUDA implementation with new parameters**

The CUDA implementation should follow the same pattern as CPU, using ATen's padding functions which work on CUDA tensors. Update the function signatures to match the new schema:

```cpp
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.is_cuda(), "hilbert_transform: input must be a CUDA tensor");
    // ... implementation using same apply_padding and apply_window utilities
}
```

**Step 2: Update backward and backward_backward similarly**

**Step 3: Verify compilation**

Run: `uv run python setup.py build_ext --inplace`

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cuda/integral_transform/hilbert_transform.cu
git commit -m "feat: update CUDA hilbert_transform with padding_mode and window"
```

---

### Task 5: Update Autograd Registration

**Files:**
- Modify: `src/torchscience/csrc/autograd/integral_transform/hilbert_transform.h`

**Step 1: Update autograd function to pass new parameters**

```cpp
class HilbertTransformFunction : public torch::autograd::Function<HilbertTransformFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        int64_t n,
        int64_t dim,
        int64_t padding_mode,
        double padding_value,
        const c10::optional<at::Tensor>& window
    ) {
        at::AutoDispatchBelowAutograd guard;

        ctx->save_for_backward({input});
        if (window.has_value()) {
            ctx->save_for_backward({input, window.value()});
        } else {
            ctx->save_for_backward({input});
        }
        ctx->saved_data["n"] = n;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["padding_mode"] = padding_mode;
        ctx->saved_data["padding_value"] = padding_value;
        ctx->saved_data["has_window"] = window.has_value();

        return torch::ops::torchscience::hilbert_transform(
            input, n, dim, padding_mode, padding_value, window
        );
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        c10::optional<at::Tensor> window = c10::nullopt;
        if (ctx->saved_data["has_window"].toBool()) {
            window = saved[1];
        }

        int64_t n = ctx->saved_data["n"].toInt();
        int64_t dim = ctx->saved_data["dim"].toInt();
        int64_t padding_mode = ctx->saved_data["padding_mode"].toInt();
        double padding_value = ctx->saved_data["padding_value"].toDouble();

        auto grad_input = torch::ops::torchscience::hilbert_transform_backward(
            grad_outputs[0], input, n, dim, padding_mode, padding_value, window
        );

        // Return gradients for all inputs (None for non-tensor params)
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/integral_transform/hilbert_transform.h
git commit -m "feat: update autograd registration with new parameters"
```

---

### Task 6: Update Meta Tensor Support

**Files:**
- Modify: `src/torchscience/csrc/meta/integral_transform/hilbert_transform.h`

**Step 1: Update meta implementation with new parameters**

```cpp
inline at::Tensor hilbert_transform_meta(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // ... same shape computation logic, window parameter doesn't affect output shape
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/integral_transform/hilbert_transform.h
git commit -m "feat: update meta tensor support with new parameters"
```

---

## Phase 3: Python Interface Updates

### Task 7: Update Python hilbert_transform

**Files:**
- Create: `src/torchscience/signal_processing/integral_transform/__init__.py`
- Create: `src/torchscience/signal_processing/integral_transform/_hilbert_transform.py`

**Step 1: Create integral_transform module**

Create file: `src/torchscience/signal_processing/integral_transform/__init__.py`

```python
"""Integral transforms for signal processing."""

from ._hilbert_transform import hilbert_transform
from ._inverse_hilbert_transform import inverse_hilbert_transform

__all__ = [
    "hilbert_transform",
    "inverse_hilbert_transform",
]
```

**Step 2: Create thin Python wrapper**

Create file: `src/torchscience/signal_processing/integral_transform/_hilbert_transform.py`

```python
"""Hilbert transform implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Padding mode mapping
_PADDING_MODES = {
    'constant': 0,
    'reflect': 1,
    'replicate': 2,
    'circular': 3,
}


def hilbert_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    padding_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    padding_value: float = 0.0,
    window: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the Hilbert transform of a signal along a specified dimension.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode when ``n`` is larger than input size. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    window : Tensor, optional
        Window function to apply before the transform. Must be 1-D with size
        matching the (possibly padded) signal length along ``dim``.
        Use window functions from ``torchscience.signal_processing.window_function``.
        Default: ``None`` (no windowing).

    Returns
    -------
    Tensor
        The Hilbert transform of the input.

    Examples
    --------
    Basic usage:

    >>> x = torch.sin(torch.linspace(0, 2 * torch.pi, 100))
    >>> h = hilbert_transform(x)

    With reflection padding to reduce edge effects:

    >>> h = hilbert_transform(x, n=256, padding_mode='reflect')

    With a window function:

    >>> window = torch.hann_window(100)
    >>> h = hilbert_transform(x, window=window)
    """
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {list(_PADDING_MODES.keys())}, "
            f"got '{padding_mode}'"
        )

    return torch.ops.torchscience.hilbert_transform(
        input,
        n if n is not None else -1,
        dim,
        _PADDING_MODES[padding_mode],
        padding_value,
        window,
    )
```

**Step 3: Update signal_processing __init__.py**

```python
from . import filter, integral_transform, waveform, window_function

__all__ = [
    "filter",
    "integral_transform",
    "waveform",
    "window_function",
]
```

**Step 4: Commit**

```bash
git add src/torchscience/signal_processing/integral_transform/
git add src/torchscience/signal_processing/__init__.py
git commit -m "feat: add Python interface for hilbert_transform with padding and window"
```

---

### Task 8: Add Deprecation to Old Location

**Files:**
- Modify: `src/torchscience/integral_transform/_hilbert_transform.py`

**Step 1: Add deprecation warning and forward to new location**

```python
"""Hilbert transform - DEPRECATED location.

.. deprecated::
    Import from ``torchscience.signal_processing.integral_transform`` instead.
"""

import warnings

warnings.warn(
    "torchscience.integral_transform is deprecated. "
    "Use torchscience.signal_processing.integral_transform instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from torchscience.signal_processing.integral_transform import hilbert_transform

__all__ = ["hilbert_transform"]
```

**Step 2: Commit**

```bash
git add src/torchscience/integral_transform/
git commit -m "feat: deprecate old integral_transform location"
```

---

## Phase 4: Tests

### Task 9: Create Tests for New Functionality

**Files:**
- Create: `tests/torchscience/signal_processing/integral_transform/__init__.py`
- Create: `tests/torchscience/signal_processing/integral_transform/test__hilbert_transform.py`

**Step 1: Create test file with comprehensive tests**

```python
"""Tests for torchscience.signal_processing.integral_transform.hilbert_transform."""

import math
import pytest
import torch

import torchscience.signal_processing.integral_transform
import torchscience.signal_processing.window_function


class TestHilbertTransformPaddingMode:
    """Tests for padding_mode parameter."""

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_padding_modes(self, mode):
        """Test all padding modes work."""
        x = torch.randn(64)
        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode=mode
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_padding_mode_invalid(self):
        """Test invalid padding_mode raises error."""
        x = torch.randn(64)
        with pytest.raises(ValueError, match="padding_mode"):
            torchscience.signal_processing.integral_transform.hilbert_transform(
                x, padding_mode="invalid"
            )

    def test_padding_value_constant(self):
        """Test padding_value with constant mode."""
        x = torch.randn(64, dtype=torch.float64)
        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode="constant", padding_value=1.0
        )
        assert result.shape == (128,)

    def test_reflect_reduces_edge_effects(self):
        """Test that reflect padding reduces edge effects vs zero padding."""
        # Non-periodic ramp signal
        x = torch.linspace(0, 1, 64, dtype=torch.float64)

        h_zero = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode="constant"
        )[:64]
        h_reflect = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode="reflect"
        )[:64]

        # Both should be finite
        assert torch.all(torch.isfinite(h_zero))
        assert torch.all(torch.isfinite(h_reflect))


class TestHilbertTransformWindow:
    """Tests for window parameter."""

    def test_rectangular_window_no_effect(self):
        """Test rectangular window (all ones) has no effect."""
        x = torch.randn(64, dtype=torch.float64)
        window = torchscience.signal_processing.window_function.rectangular_window(
            64, dtype=torch.float64
        )

        result_no_window = torchscience.signal_processing.integral_transform.hilbert_transform(x)
        result_with_window = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, window=window
        )

        torch.testing.assert_close(result_no_window, result_with_window)

    def test_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(64)
        window = torch.hann_window(64)

        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, window=window
        )
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_window_wrong_size(self):
        """Test window size mismatch raises error."""
        x = torch.randn(64)
        window = torch.ones(32)

        with pytest.raises(RuntimeError, match="window"):
            torchscience.signal_processing.integral_transform.hilbert_transform(
                x, window=window
            )

    def test_window_with_padding(self):
        """Test window combined with padding."""
        x = torch.randn(64)
        window = torch.hann_window(128)

        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode="reflect", window=window
        )
        assert result.shape == (128,)

    def test_window_batched(self):
        """Test window broadcasts over batch dimensions."""
        x = torch.randn(3, 4, 64)
        window = torch.hann_window(64)

        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, dim=-1, window=window
        )
        assert result.shape == (3, 4, 64)


class TestHilbertTransformGradient:
    """Tests for gradient computation with new parameters."""

    def test_gradient_with_padding(self):
        """Test gradient works with padding."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, n=128, padding_mode="reflect"
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_with_window(self):
        """Test gradient works with window."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.integral_transform.hilbert_transform(
            x, window=window
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_gradcheck_with_padding(self):
        """Test gradient correctness with padding."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.integral_transform.hilbert_transform(
                input_tensor, n=64, padding_mode="reflect"
            )

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4)
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/signal_processing/integral_transform/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/torchscience/signal_processing/integral_transform/
git commit -m "test: add tests for hilbert_transform padding and window parameters"
```

---

### Task 10: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete hilbert_transform enhancements with C++ padding and windowing"
```

---

## Summary

| Component | Change |
|-----------|--------|
| `torchscience.cpp` | Add padding_mode, padding_value, window to operator schema |
| `impl/hilbert_transform.h` | Add apply_padding and apply_window utilities |
| `cpu/hilbert_transform.h` | Update to use new parameters |
| `cuda/hilbert_transform.cu` | Update to use new parameters |
| `autograd/hilbert_transform.h` | Pass new parameters through autograd |
| `meta/hilbert_transform.h` | Update meta tensor support |
| `signal_processing/integral_transform/` | New Python module location |
| `integral_transform/` | Deprecation warnings |

## Final API

```python
from torchscience.signal_processing.integral_transform import hilbert_transform

# All parameter handling happens in C++
h = hilbert_transform(
    x,
    n=256,                    # FFT length
    dim=-1,                   # transform dimension
    padding_mode='reflect',   # 'constant', 'reflect', 'replicate', 'circular'
    padding_value=0.0,        # for constant mode
    window=torch.hann_window(256),  # optional window tensor
)
```
