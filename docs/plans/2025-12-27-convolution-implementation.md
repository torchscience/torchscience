# Convolution Operator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.signal_processing.convolution` as a differentiable N-dimensional convolution operator with adaptive algorithm selection (FFT vs direct).

**Architecture:** C++ backend following the butterworth filter pattern. Forward computes N-D convolution using either FFT or direct sliding-window based on kernel size. Backward computes gradients via convolution with flipped kernel. Supports batch broadcasting over leading dimensions.

**Tech Stack:** C++17, PyTorch C++ API (ATen), torch.fft, torch.autograd

### Design Decisions (from Brainstorming)

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Dimensions** | N-D with batching | Maximum flexibility for signals, images, volumes |
| **Modes** | full, same, valid | Standard convolution modes (scipy-compatible) |
| **Algorithm** | Adaptive (FFT/direct) | FFT faster for large kernels, direct for small |
| **Batching** | Leading batch dims | input: (*batch, *spatial), kernel: (*spatial) |
| **Boundary** | Configurable padding | zero, wrap, reflect, constant value |
| **Autograd** | Second-order gradients | Support for higher-order optimization |

## API Specification

```python
def convolution(
    input: Tensor,           # (*batch, *spatial) - signal to convolve
    kernel: Tensor,          # (*spatial) - convolution kernel
    mode: str = "full",      # "full", "same", or "valid"
    *,
    padding_mode: str = "zero",     # "zero", "wrap", "reflect", "constant"
    padding_value: float = 0.0,     # value for padding_mode="constant"
) -> Tensor:
    """
    N-dimensional convolution with automatic algorithm selection.

    Parameters
    ----------
    input : Tensor, shape (*batch, *spatial)
        Input signal. Last N dimensions are spatial dimensions that match
        the kernel's dimensionality.
    kernel : Tensor, shape (*spatial)
        Convolution kernel. Must have same number of dimensions as input's
        spatial dimensions.
    mode : str, default="full"
        Output size mode:
        - "full": Full convolution output (n + k - 1)
        - "same": Output same size as input (n)
        - "valid": Only valid overlap positions (n - k + 1)
    padding_mode : str, default="zero"
        How to handle boundaries for "same" mode:
        - "zero": Pad with zeros
        - "wrap": Circular/periodic padding
        - "reflect": Mirror padding at boundaries
        - "constant": Pad with padding_value
    padding_value : float, default=0.0
        Value for constant padding mode.

    Returns
    -------
    Tensor, shape (*batch, *output_spatial)
        Convolved output. Batch dimensions preserved, spatial dimensions
        determined by mode.

    Examples
    --------
    1D convolution:

    >>> signal = torch.randn(100)
    >>> kernel = torch.tensor([1., 2., 1.]) / 4
    >>> output = torchscience.signal_processing.convolution(signal, kernel)
    >>> output.shape
    torch.Size([102])  # 100 + 3 - 1

    2D convolution with batching:

    >>> images = torch.randn(32, 64, 64)  # batch of 32 images
    >>> blur_kernel = torch.ones(5, 5) / 25
    >>> blurred = torchscience.signal_processing.convolution(
    ...     images, blur_kernel, mode="same"
    ... )
    >>> blurred.shape
    torch.Size([32, 64, 64])

    Notes
    -----
    - Uses FFT convolution for large kernels (numel > 32) for O(n log n) complexity
    - Uses direct convolution for small kernels for lower overhead
    - Gradients are computed via convolution with flipped kernel
    - Supports float16, bfloat16, float32, float64 dtypes
    """
```

---

## Mathematical Background

### Discrete Convolution

For N-dimensional discrete convolution:

$$(f * g)[n_1, ..., n_N] = \sum_{m_1} \cdots \sum_{m_N} f[m_1, ..., m_N] \cdot g[n_1 - m_1, ..., n_N - m_N]$$

### Output Size by Mode

For input of size $n$ and kernel of size $k$ in each dimension:

| Mode | Output Size | Description |
|------|-------------|-------------|
| `full` | $n + k - 1$ | All positions where kernel overlaps input |
| `same` | $n$ | Center output on input, requires padding |
| `valid` | $n - k + 1$ | Only positions with full overlap |

### FFT Convolution

Using the convolution theorem:

$$f * g = \mathcal{F}^{-1}\{\mathcal{F}\{f\} \cdot \mathcal{F}\{g\}\}$$

Where $\mathcal{F}$ is the N-D FFT. Complexity: $O(N \log N)$ vs $O(N \cdot K)$ for direct.

### Gradient Computation

For $y = x * k$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} * \text{flip}(k)$$

$$\frac{\partial L}{\partial k} = \text{correlate}(x, \frac{\partial L}{\partial y})$$

---

## Task 1: Create signal_processing.convolution module Python structure

**Files:**
- Create: `src/torchscience/signal_processing/convolution/__init__.py`
- Create: `src/torchscience/signal_processing/convolution/_convolution.py`
- Modify: `src/torchscience/signal_processing/__init__.py`

**Step 1: Create the convolution submodule __init__.py**

```python
# src/torchscience/signal_processing/convolution/__init__.py
from ._convolution import convolution

__all__ = [
    "convolution",
]
```

**Step 2: Create the Python API**

```python
# src/torchscience/signal_processing/convolution/_convolution.py
"""N-dimensional convolution implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def convolution(
    input: Tensor,
    kernel: Tensor,
    mode: str = "full",
    *,
    padding_mode: str = "zero",
    padding_value: float = 0.0,
) -> Tensor:
    r"""Compute N-dimensional convolution with automatic algorithm selection.

    Computes the discrete convolution of an N-dimensional input signal with
    a kernel. Automatically selects between FFT-based convolution (faster for
    large kernels) and direct sliding-window convolution (faster for small
    kernels).

    Mathematical Definition
    -----------------------
    For N-dimensional discrete convolution:

    .. math::
        (f * g)[n_1, \ldots, n_N] = \sum_{m_1} \cdots \sum_{m_N}
        f[m_1, \ldots, m_N] \cdot g[n_1 - m_1, \ldots, n_N - m_N]

    Parameters
    ----------
    input : Tensor, shape (*batch, *spatial)
        Input signal to convolve. The last ``kernel.ndim`` dimensions are
        treated as spatial dimensions; any preceding dimensions are batch
        dimensions that are broadcast over.
    kernel : Tensor, shape (*spatial)
        Convolution kernel. Must have at least 1 dimension and no more
        dimensions than ``input``.
    mode : str, default="full"
        Specifies the output size:

        - ``"full"``: Full discrete convolution. Output size is
          ``input_size + kernel_size - 1`` in each spatial dimension.
        - ``"same"``: Output has same size as input in spatial dimensions.
          Requires padding at boundaries.
        - ``"valid"``: Output contains only positions where kernel fully
          overlaps input. Output size is ``input_size - kernel_size + 1``.
    padding_mode : str, default="zero"
        How to handle boundaries for ``"same"`` mode:

        - ``"zero"``: Pad with zeros (default)
        - ``"wrap"``: Circular/periodic padding (wraps around)
        - ``"reflect"``: Reflect values at boundaries
        - ``"constant"``: Pad with ``padding_value``
    padding_value : float, default=0.0
        Value to use when ``padding_mode="constant"``.

    Returns
    -------
    Tensor
        Convolved output with shape ``(*batch, *output_spatial)``, where
        ``output_spatial`` depends on ``mode``.

    Examples
    --------
    1D convolution (smoothing filter):

    >>> signal = torch.randn(100)
    >>> kernel = torch.tensor([1., 2., 1.]) / 4  # Gaussian-like
    >>> smoothed = torchscience.signal_processing.convolution(signal, kernel)
    >>> smoothed.shape
    torch.Size([102])  # full: 100 + 3 - 1

    Same-size output:

    >>> smoothed = torchscience.signal_processing.convolution(
    ...     signal, kernel, mode="same"
    ... )
    >>> smoothed.shape
    torch.Size([100])

    2D convolution with batching:

    >>> images = torch.randn(32, 64, 64)  # batch of 32 64x64 images
    >>> blur = torch.ones(5, 5) / 25  # box blur
    >>> blurred = torchscience.signal_processing.convolution(
    ...     images, blur, mode="same"
    ... )
    >>> blurred.shape
    torch.Size([32, 64, 64])

    With gradients:

    >>> x = torch.randn(10, requires_grad=True)
    >>> k = torch.randn(3, requires_grad=True)
    >>> y = torchscience.signal_processing.convolution(x, k)
    >>> y.sum().backward()
    >>> x.grad.shape, k.grad.shape
    (torch.Size([10]), torch.Size([3]))

    Notes
    -----
    **Algorithm Selection:**

    - Kernel numel > 32: Uses FFT convolution, O(n log n) complexity
    - Kernel numel <= 32: Uses direct convolution, lower overhead

    **Gradient Support:**

    First and second-order gradients are supported. Gradients are computed
    via convolution with the flipped kernel.

    **Dtype Support:**

    Supports float16, bfloat16, float32, and float64. Half-precision types
    are computed in float32 internally for numerical stability.

    See Also
    --------
    scipy.signal.convolve : SciPy's N-dimensional convolution
    scipy.ndimage.convolve : SciPy's N-dimensional convolution with more padding options
    torch.nn.functional.conv1d : PyTorch's 1D convolution (different convention)
    """
    # Input validation
    if kernel.ndim == 0:
        raise ValueError("kernel must have at least 1 dimension")
    if kernel.ndim > input.ndim:
        raise ValueError(
            f"kernel has more dimensions ({kernel.ndim}) than input ({input.ndim})"
        )
    if mode not in ("full", "same", "valid"):
        raise ValueError(
            f"mode must be 'full', 'same', or 'valid', got '{mode}'"
        )
    if padding_mode not in ("zero", "wrap", "reflect", "constant"):
        raise ValueError(
            f"padding_mode must be 'zero', 'wrap', 'reflect', or 'constant', "
            f"got '{padding_mode}'"
        )

    # Check spatial dimensions match
    spatial_ndim = kernel.ndim
    input_spatial = input.shape[-spatial_ndim:]
    for i, (in_size, k_size) in enumerate(zip(input_spatial, kernel.shape)):
        if mode == "valid" and in_size < k_size:
            raise ValueError(
                f"For mode='valid', input spatial size ({in_size}) must be >= "
                f"kernel size ({k_size}) in dimension {i}"
            )

    # Convert padding_mode to integer for C++
    padding_mode_int = {"zero": 0, "wrap": 1, "reflect": 2, "constant": 3}[padding_mode]

    # Ensure compatible dtypes and devices
    target_dtype = torch.promote_types(input.dtype, kernel.dtype)
    if input.dtype != target_dtype:
        input = input.to(target_dtype)
    if kernel.dtype != target_dtype:
        kernel = kernel.to(target_dtype)
    if kernel.device != input.device:
        kernel = kernel.to(input.device)

    return torch.ops.torchscience.convolution(
        input, kernel, mode, padding_mode_int, padding_value
    )
```

**Step 3: Update signal_processing __init__.py**

Add `convolution` to the imports in `src/torchscience/signal_processing/__init__.py`.

**Step 4: Commit**

```bash
git add src/torchscience/signal_processing/convolution/
git add src/torchscience/signal_processing/__init__.py
git commit -m "feat(signal_processing): add convolution module Python structure"
```

---

## Task 2: Create impl header with forward algorithm

**Files:**
- Create: `src/torchscience/csrc/impl/signal_processing/convolution/convolution.h`

**Step 1: Create forward implementation**

```cpp
// src/torchscience/csrc/impl/signal_processing/convolution/convolution.h
#pragma once

/*
 * N-dimensional Convolution Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * For N-dimensional discrete convolution:
 *
 *   (f * g)[n₁, ..., nₙ] = Σ_{m₁} ... Σ_{mₙ} f[m₁, ..., mₙ] · g[n₁-m₁, ..., nₙ-mₙ]
 *
 * OUTPUT SIZE BY MODE:
 * ====================
 *   full:  input_size + kernel_size - 1
 *   same:  input_size
 *   valid: input_size - kernel_size + 1
 *
 * ALGORITHM SELECTION:
 * ====================
 *   kernel_numel > threshold: FFT convolution O(n log n)
 *   kernel_numel <= threshold: Direct convolution O(n * k)
 */

#include <c10/macros/Macros.h>
#include <cmath>
#include <algorithm>

namespace torchscience::impl::signal_processing::convolution {

// Threshold for switching between direct and FFT convolution
constexpr int64_t FFT_THRESHOLD = 32;

// Padding mode enum
enum class PaddingMode : int {
    Zero = 0,
    Wrap = 1,
    Reflect = 2,
    Constant = 3
};

/**
 * Compute 1D direct convolution for a single pair of input/kernel.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void convolve_1d_direct(
    const T* input,
    int64_t input_size,
    const T* kernel,
    int64_t kernel_size,
    T* output,
    int64_t output_size,
    int64_t output_offset  // offset from full output start
) {
    for (int64_t i = 0; i < output_size; ++i) {
        T sum = T(0);
        int64_t out_idx = i + output_offset;
        for (int64_t j = 0; j < kernel_size; ++j) {
            int64_t in_idx = out_idx - j;
            if (in_idx >= 0 && in_idx < input_size) {
                sum += input[in_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }
}

/**
 * Compute output size for given mode.
 */
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t compute_output_size(
    int64_t input_size,
    int64_t kernel_size,
    int mode  // 0=full, 1=same, 2=valid
) {
    switch (mode) {
        case 0:  // full
            return input_size + kernel_size - 1;
        case 1:  // same
            return input_size;
        case 2:  // valid
            return std::max(int64_t(0), input_size - kernel_size + 1);
        default:
            return input_size + kernel_size - 1;
    }
}

/**
 * Compute output offset in full output for given mode.
 */
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t compute_output_offset(
    int64_t kernel_size,
    int mode  // 0=full, 1=same, 2=valid
) {
    switch (mode) {
        case 0:  // full
            return 0;
        case 1:  // same
            return (kernel_size - 1) / 2;
        case 2:  // valid
            return kernel_size - 1;
        default:
            return 0;
    }
}

}  // namespace torchscience::impl::signal_processing::convolution
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/signal_processing/convolution/
git commit -m "feat(impl): add convolution forward implementation"
```

---

## Task 3: Create impl header with backward algorithm

**Files:**
- Create: `src/torchscience/csrc/impl/signal_processing/convolution/convolution_backward.h`

**Step 1: Create backward implementation**

The backward pass for convolution:
- grad_input = convolve(grad_output, flip(kernel), mode adjusted)
- grad_kernel = correlate(input, grad_output)

```cpp
// src/torchscience/csrc/impl/signal_processing/convolution/convolution_backward.h
#pragma once

#include <c10/macros/Macros.h>
#include "convolution.h"

namespace torchscience::impl::signal_processing::convolution {

/**
 * Flip a 1D kernel for gradient computation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void flip_kernel_1d(
    const T* kernel,
    int64_t kernel_size,
    T* flipped
) {
    for (int64_t i = 0; i < kernel_size; ++i) {
        flipped[i] = kernel[kernel_size - 1 - i];
    }
}

/**
 * Compute 1D correlation (convolution with flipped kernel).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void correlate_1d_direct(
    const T* input,
    int64_t input_size,
    const T* kernel,
    int64_t kernel_size,
    T* output,
    int64_t output_size
) {
    // Correlation is convolution with the kernel NOT flipped
    // For gradient w.r.t. kernel, we correlate input with grad_output
    for (int64_t i = 0; i < output_size; ++i) {
        T sum = T(0);
        for (int64_t j = 0; j < kernel_size; ++j) {
            int64_t in_idx = i + j;
            if (in_idx >= 0 && in_idx < input_size) {
                sum += input[in_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }
}

}  // namespace torchscience::impl::signal_processing::convolution
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/impl/signal_processing/convolution/convolution_backward.h
git commit -m "feat(impl): add convolution backward implementation"
```

---

## Task 4: Create CPU kernel with FFT and direct paths

**Files:**
- Create: `src/torchscience/csrc/cpu/signal_processing/convolution.h`

**Step 1: Create CPU implementation**

The CPU kernel should:
- Use `AT_DISPATCH_FLOATING_TYPES_AND2` for dtype dispatch
- Use `at::parallel_for` for batch parallelization
- Implement both direct and FFT paths
- Handle N-dimensional convolution via separable 1D passes or full N-D FFT
- Register with `TORCH_LIBRARY_IMPL(torchscience, CPU, module)`

Key implementation notes:
- For FFT path: use `torch::fft::fftn` and `torch::fft::ifftn`
- Pad both input and kernel to `input_size + kernel_size - 1` for full convolution
- Trim output based on mode

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/signal_processing/convolution.h
git commit -m "feat(cpu): add convolution CPU kernel with FFT/direct selection"
```

---

## Task 5: Create autograd wrapper

**Files:**
- Create: `src/torchscience/csrc/autograd/signal_processing/convolution.h`

**Step 1: Create autograd Function class**

The autograd wrapper should:
- Create `Convolution` class inheriting from `torch::autograd::Function`
- Save input, kernel, and mode for backward
- Implement backward using convolution with flipped kernel
- Support double-backward for second-order gradients
- Register with `TORCH_LIBRARY_IMPL(torchscience, Autograd, module)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/signal_processing/convolution.h
git commit -m "feat(autograd): add convolution autograd wrapper"
```

---

## Task 6: Create meta tensor implementation

**Files:**
- Create: `src/torchscience/csrc/meta/signal_processing/convolution.h`

**Step 1: Create meta implementation for shape inference**

The meta implementation should:
- Compute output shape based on input shape, kernel shape, and mode
- Handle batch dimensions
- Register with `TORCH_LIBRARY_IMPL(torchscience, Meta, m)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/signal_processing/convolution.h
git commit -m "feat(meta): add convolution meta tensor implementation"
```

---

## Task 7: Create autocast wrapper

**Files:**
- Create: `src/torchscience/csrc/autocast/signal_processing/convolution.h`

**Step 1: Create autocast implementation**

The autocast wrapper should:
- Handle mixed precision by casting to appropriate dtype
- Register with `TORCH_LIBRARY_IMPL(torchscience, Autocast, m)`

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autocast/signal_processing/convolution.h
git commit -m "feat(autocast): add convolution autocast wrapper"
```

---

## Task 8: Register operators in torchscience.cpp

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add operator schema definitions**

Add to the `TORCH_LIBRARY(torchscience, module)` block:

```cpp
// signal_processing.convolution
module.def("convolution(Tensor input, Tensor kernel, str mode, int padding_mode, float padding_value) -> Tensor");
module.def("convolution_backward(Tensor grad_output, Tensor input, Tensor kernel, str mode, int padding_mode, float padding_value) -> (Tensor, Tensor)");
```

**Step 2: Add includes for the new headers**

```cpp
#include "cpu/signal_processing/convolution.h"
#include "autograd/signal_processing/convolution.h"
#include "meta/signal_processing/convolution.h"
#include "autocast/signal_processing/convolution.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register convolution operators in TORCH_LIBRARY"
```

---

## Task 9: Write comprehensive tests

**Files:**
- Create: `tests/torchscience/signal_processing/convolution/__init__.py`
- Create: `tests/torchscience/signal_processing/convolution/test__convolution.py`

**Step 1: Create test file**

Tests should cover:

1. **Basic functionality**
   - Output shapes for each mode (full, same, valid)
   - 1D, 2D, 3D convolution
   - Batched convolution

2. **Correctness (compare to scipy.signal.convolve)**
   - Identity kernel [1] returns input
   - Shift kernel [0, 1, 0] shifts input
   - Box filter matches expected smoothing
   - Compare to scipy.signal.convolve for all modes

3. **Algorithm selection**
   - Small kernel uses direct
   - Large kernel uses FFT
   - Both produce same results

4. **Gradients**
   - `torch.autograd.gradcheck` passes
   - Second-order gradients work

5. **Dtype support**
   - float32, float64
   - float16, bfloat16 (if supported)

6. **Edge cases**
   - Single-element input
   - Single-element kernel
   - Kernel larger than input (valid mode returns empty)

**Step 2: Commit**

```bash
git add tests/torchscience/signal_processing/convolution/
git commit -m "test: add comprehensive tests for convolution"
```

---

## Task 10: Build and verify

**Step 1: Build the project**

```bash
uv run pip install -e .
```

**Step 2: Run all convolution tests**

```bash
uv run pytest tests/torchscience/signal_processing/convolution/test__convolution.py -v
```

**Step 3: Verify import works**

```bash
uv run python -c "from torchscience.signal_processing.convolution import convolution; print(convolution)"
```

**Step 4: Verify scipy comparison**

```bash
uv run python -c "
import torch
import numpy as np
from scipy.signal import convolve as scipy_convolve
from torchscience.signal_processing.convolution import convolution

x = torch.randn(100)
k = torch.randn(10)

for mode in ['full', 'same', 'valid']:
    ts_result = convolution(x, k, mode=mode)
    sp_result = scipy_convolve(x.numpy(), k.numpy(), mode=mode)
    diff = torch.abs(ts_result - torch.from_numpy(sp_result)).max()
    print(f'{mode}: max diff = {diff:.2e}')
"
```

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: address any build or test issues"
```

---

## Summary

This plan implements `torchscience.signal_processing.convolution` as a differentiable N-dimensional convolution operator:

1. **Python API** - User-facing function with validation, mode selection, and padding options
2. **impl headers** - Device-agnostic algorithms for direct and FFT convolution
3. **CPU kernel** - Adaptive algorithm selection with parallel batch processing
4. **Autograd wrapper** - First and second-order gradient support
5. **Meta implementation** - Shape inference for torch.compile
6. **Autocast wrapper** - Mixed precision support
7. **Operator registration** - TORCH_LIBRARY schema definitions
8. **Tests** - Comprehensive correctness, gradient, and scipy comparison tests

**Key features:**
- N-dimensional support with batch broadcasting
- Adaptive FFT/direct algorithm selection
- Three output modes: full, same, valid
- Configurable boundary padding
- Full autograd support including second-order gradients
- Scipy-compatible behavior

**References:**
- [SciPy signal.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html)
- [NumPy convolve](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
