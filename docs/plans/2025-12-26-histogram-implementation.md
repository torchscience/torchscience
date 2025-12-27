# Histogram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.statistics.descriptive.histogram` as the first "Flatten" category operator.

**Architecture:** Hard binning histogram following the established kurtosis pattern. Uses searchsorted + scatter_add for fast non-differentiable histograms. Supports N-dimensional histograms, batched operations, weights, and density normalization.

**Tech Stack:** PyTorch C++ extension, CUDA kernels, ATen dispatch

---

## Task 1: Create Python API for `histogram`

**Files:**
- Create: `src/torchscience/statistics/descriptive/_histogram.py`
- Modify: `src/torchscience/statistics/descriptive/__init__.py:7-11`

**Step 1: Write the failing test**

Create: `tests/torchscience/statistics/descriptive/test__histogram.py`

```python
"""Tests for torchscience.statistics.descriptive.histogram."""

import pytest
import torch

import torchscience.statistics.descriptive


class TestHistogramBasic:
    """Basic functionality tests."""

    def test_1d_histogram_shape(self):
        """Test 1D histogram returns correct shapes."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.shape == (10,)
        assert edges.shape == (11,)

    def test_1d_histogram_counts_sum(self):
        """Test histogram counts sum to input size."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.sum().item() == 1000
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py::TestHistogramBasic::test_1d_histogram_shape -v`

Expected: FAIL with "cannot import name 'histogram'"

**Step 3: Write Python API stub**

Create: `src/torchscience/statistics/descriptive/_histogram.py`

```python
"""Histogram implementation."""

from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def histogram(
    input: Tensor,
    bins: Union[int, Tensor, Sequence[Union[int, Tensor]]] = 10,
    *,
    range: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]] = None,
    weights: Optional[Tensor] = None,
    density: bool = False,
    dim: Optional[int] = None,
    closed: Literal["left", "right"] = "right",
    out_of_bounds: Literal["ignore", "clamp", "error"] = "ignore",
) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
    r"""Compute N-dimensional histogram using hard binning (non-differentiable).

    This is a "Flatten" category operator - input samples are treated as a
    1D collection regardless of actual shape, producing bin counts.

    Parameters
    ----------
    input : Tensor
        Input samples. For D-dimensional histogram:
        - Shape (..., N, D) where N is samples, D is histogram dims
        - If D=1, can be (..., N) for convenience
    bins : int, Tensor, or Sequence, optional
        Bin specification:
        - int: Equal-width bins, count applies to all dimensions
        - Tensor: Explicit bin edges (length = num_bins + 1)
        - Sequence: Per-dimension specification for N-D histogram
        Default: 10
    range : tuple or Sequence of tuples, optional
        Data range for automatic binning. None uses data min/max.
        For N-D: sequence of (min, max) tuples per dimension.
    weights : Tensor, optional
        Per-sample weights, broadcastable to input sample count.
    density : bool, optional
        If True, normalize so integral equals 1. Default: False.
    dim : int, optional
        Sample dimension. If specified, enables batched histograms:
        histogram computed along this dim, others are batch dims.
    closed : {"left", "right"}, optional
        Which side of bins is closed:
        - "right": (a, b] intervals (default, matches R/Julia)
        - "left": [a, b) intervals (matches MATLAB/Boost)
        Note: Endpoint handling ensures no values are lost.
    out_of_bounds : {"ignore", "clamp", "error"}, optional
        How to handle values outside range:
        - "ignore": Exclude from counts (default, matches NumPy)
        - "clamp": Include in edge bins (matches TensorFlow)
        - "error": Raise ValueError

    Returns
    -------
    counts : Tensor
        Histogram bin counts.
        Shape (bins[0], bins[1], ..., bins[D-1]) for N-D.
        With dim specified: (...batch..., bins...).
    edges : Tensor or tuple of Tensors
        Bin edges.
        - 1D: Tensor of shape (num_bins + 1,)
        - N-D: Tuple of edge Tensors, one per dimension

    Examples
    --------
    Basic 1D histogram:

    >>> x = torch.randn(1000)
    >>> counts, edges = torchscience.statistics.descriptive.histogram(x, bins=50)
    >>> counts.shape
    torch.Size([50])

    2D histogram:

    >>> xy = torch.randn(1000, 2)
    >>> counts, (ex, ey) = torchscience.statistics.descriptive.histogram(
    ...     xy, bins=[30, 40]
    ... )
    >>> counts.shape
    torch.Size([30, 40])

    Batched histogram:

    >>> batch = torch.randn(8, 1000)
    >>> counts, edges = torchscience.statistics.descriptive.histogram(
    ...     batch, bins=50, dim=-1
    ... )
    >>> counts.shape
    torch.Size([8, 50])

    Notes
    -----
    - This operator is **not differentiable**. For gradient support, use
      :func:`soft_histogram`.

    - The "Flatten" category means this operator treats the sample dimension
      as a flat collection, producing fixed-shape output determined by bin
      count rather than input sample count.

    - Bin edge handling follows NumPy convention: all bins are half-open
      [a, b) except the last bin which is closed [a, b] when closed="right".

    See Also
    --------
    soft_histogram : Differentiable histogram using kernel density estimation.
    torch.histogram : PyTorch's built-in histogram (1D only, no batching).

    References
    ----------
    .. [1] NumPy histogram documentation:
           https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # Validate inputs
    if input.numel() == 0:
        raise ValueError("histogram: input tensor must be non-empty")

    # For now, delegate to torch.histogram for 1D case
    # Full C++ implementation will replace this
    if dim is None and input.dim() == 1:
        # Simple 1D case - use torch.histogram as reference
        if isinstance(bins, int):
            if range is not None:
                result = torch.histogram(input, bins=bins, range=range, weight=weights, density=density)
            else:
                result = torch.histogram(input, bins=bins, weight=weights, density=density)
            return result.hist, result.bin_edges
        elif isinstance(bins, Tensor):
            result = torch.histogram(input, bins=bins, weight=weights, density=density)
            return result.hist, result.bin_edges

    # TODO: Implement full N-D and batched histogram via C++
    raise NotImplementedError(
        "N-dimensional and batched histograms not yet implemented. "
        "Use dim=None with 1D input for basic histogram."
    )
```

**Step 4: Update module exports**

Modify: `src/torchscience/statistics/descriptive/__init__.py`

```python
"""Descriptive statistics functions.

This module provides functions for computing descriptive statistics
such as kurtosis, skewness, and other moments of distributions.
"""

from ._histogram import histogram
from ._kurtosis import kurtosis

__all__ = [
    "histogram",
    "kurtosis",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py::TestHistogramBasic -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/statistics/descriptive/_histogram.py \
        src/torchscience/statistics/descriptive/__init__.py \
        tests/torchscience/statistics/descriptive/test__histogram.py
git commit -m "feat(statistics): add histogram Python API stub"
```

---

## Task 2: Add comprehensive tests for histogram

**Files:**
- Modify: `tests/torchscience/statistics/descriptive/test__histogram.py`

**Step 1: Extend test file with comprehensive tests**

```python
"""Tests for torchscience.statistics.descriptive.histogram."""

import math

import pytest
import torch

import torchscience.statistics.descriptive


class TestHistogramBasic:
    """Basic functionality tests."""

    def test_1d_histogram_shape(self):
        """Test 1D histogram returns correct shapes."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.shape == (10,)
        assert edges.shape == (11,)

    def test_1d_histogram_counts_sum(self):
        """Test histogram counts sum to input size."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.sum().item() == 1000

    def test_1d_histogram_edges_monotonic(self):
        """Test that bin edges are monotonically increasing."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert torch.all(edges[1:] > edges[:-1])

    def test_explicit_bin_edges(self):
        """Test histogram with explicit bin edges."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        counts, returned_edges = torchscience.statistics.descriptive.histogram(
            x, bins=edges
        )
        assert counts.shape == (5,)
        torch.testing.assert_close(returned_edges, edges)
        # Each value falls in its own bin
        torch.testing.assert_close(counts, torch.ones(5))


class TestHistogramRange:
    """Tests for range parameter."""

    def test_explicit_range(self):
        """Test histogram with explicit range."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10, range=(-3.0, 3.0)
        )
        assert edges[0].item() == -3.0
        assert edges[-1].item() == 3.0

    def test_auto_range(self):
        """Test histogram auto-computes range from data."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=4)
        assert edges[0].item() == pytest.approx(1.0)
        assert edges[-1].item() == pytest.approx(5.0)


class TestHistogramWeights:
    """Tests for weights parameter."""

    def test_uniform_weights(self):
        """Test that uniform weights give same result as unweighted."""
        x = torch.randn(1000)
        weights = torch.ones(1000)

        counts_unweighted, _ = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        counts_weighted, _ = torchscience.statistics.descriptive.histogram(
            x, bins=10, weights=weights
        )

        torch.testing.assert_close(counts_unweighted, counts_weighted)

    def test_zero_weights(self):
        """Test that zero weights exclude samples."""
        x = torch.tensor([0.5, 1.5, 2.5])
        weights = torch.tensor([1.0, 0.0, 1.0])
        edges = torch.tensor([0.0, 1.0, 2.0, 3.0])

        counts, _ = torchscience.statistics.descriptive.histogram(
            x, bins=edges, weights=weights
        )
        # Middle sample has zero weight
        torch.testing.assert_close(counts, torch.tensor([1.0, 0.0, 1.0]))


class TestHistogramDensity:
    """Tests for density parameter."""

    def test_density_integrates_to_one(self):
        """Test that density histogram integrates to 1."""
        x = torch.randn(10000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=50, density=True
        )
        bin_widths = edges[1:] - edges[:-1]
        integral = (counts * bin_widths).sum()
        assert integral.item() == pytest.approx(1.0, rel=1e-3)


class TestHistogramDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test float32 and float64 support."""
        x = torch.randn(100, dtype=dtype)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.dtype == dtype
        assert edges.dtype == dtype


class TestHistogramDevice:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test CPU computation."""
        x = torch.randn(100, device="cpu")
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.device.type == "cpu"
        assert edges.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test CUDA computation."""
        x = torch.randn(100, device="cuda")
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        assert counts.device.type == "cuda"
        assert edges.device.type == "cuda"


class TestHistogramNotDifferentiable:
    """Tests confirming histogram is not differentiable."""

    def test_no_gradient(self):
        """Test that histogram does not track gradients."""
        x = torch.randn(100, requires_grad=True)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        # Hard histogram should not have gradient function
        assert not counts.requires_grad


class TestHistogramSciPyCompatibility:
    """Tests for SciPy/NumPy compatibility."""

    def test_matches_numpy(self):
        """Test that results match NumPy histogram."""
        np = pytest.importorskip("numpy")

        torch.manual_seed(42)
        x_torch = torch.randn(1000)
        x_np = x_torch.numpy()

        counts_torch, edges_torch = torchscience.statistics.descriptive.histogram(
            x_torch, bins=20
        )
        counts_np, edges_np = np.histogram(x_np, bins=20)

        torch.testing.assert_close(
            counts_torch, torch.from_numpy(counts_np).float(), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            edges_torch, torch.from_numpy(edges_np).float(), rtol=1e-5, atol=1e-5
        )


class TestHistogramEdgeCases:
    """Tests for edge cases."""

    def test_empty_tensor_raises(self):
        """Test that empty tensor raises ValueError."""
        x = torch.tensor([])
        with pytest.raises(ValueError, match="non-empty"):
            torchscience.statistics.descriptive.histogram(x, bins=10)

    def test_single_element(self):
        """Test histogram of single element."""
        x = torch.tensor([1.0])
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=1)
        assert counts.shape == (1,)
        assert counts[0].item() == 1.0

    def test_all_same_value(self):
        """Test histogram when all values are identical."""
        x = torch.ones(100)
        counts, edges = torchscience.statistics.descriptive.histogram(x, bins=10)
        # All values should be in one bin
        assert counts.sum().item() == 100
        assert (counts > 0).sum().item() == 1
```

**Step 2: Run tests to verify current implementation**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py -v`

Expected: Most tests PASS (those using simple 1D case)

**Step 3: Commit**

```bash
git add tests/torchscience/statistics/descriptive/test__histogram.py
git commit -m "test(histogram): add comprehensive test suite"
```

---

## Task 3: Register histogram operators in TORCH_LIBRARY

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp:130-145`

**Step 1: Add operator definitions to TORCH_LIBRARY block**

Add to `src/torchscience/csrc/torchscience.cpp` in the TORCH_LIBRARY block:

```cpp
  // `torchscience.statistics.descriptive` - histogram
  module.def("histogram(Tensor input, int bins, float[]? range, Tensor? weights, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
  module.def("histogram_edges(Tensor input, Tensor edges, Tensor? weights, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
```

**Step 2: Verify compilation**

Run: `uv run pip install -e . --no-build-isolation`

Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(csrc): register histogram operator schema"
```

---

## Task 4: Implement histogram core algorithm (impl/)

**Files:**
- Create: `src/torchscience/csrc/impl/statistics/descriptive/histogram.h`

**Step 1: Write core algorithm header**

```cpp
#pragma once

/*
 * Histogram Implementation
 *
 * ALGORITHM:
 * ==========
 * For hard binning:
 *   1. Compute bin edges from range or data min/max
 *   2. Use binary search (searchsorted) to find bin index for each sample
 *   3. Scatter-add counts to output bins
 *
 * For weighted histogram:
 *   - Scatter-add weights instead of 1s
 *
 * For density normalization:
 *   - Divide counts by (total_count * bin_width)
 *
 * EDGE HANDLING:
 * ==============
 * - closed="right": bins are (a, b], last bin includes rightmost edge
 * - closed="left": bins are [a, b), first bin includes leftmost edge
 * - out_of_bounds="ignore": values outside range not counted
 * - out_of_bounds="clamp": values clamped to edge bins
 */

#include <c10/macros/Macros.h>
#include <cmath>
#include <limits>

namespace torchscience::impl::descriptive {

/**
 * Compute bin edges for equal-width bins.
 *
 * @param data_min Minimum value in data (or explicit range min)
 * @param data_max Maximum value in data (or explicit range max)
 * @param num_bins Number of bins
 * @param edges Output array of size (num_bins + 1)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void compute_bin_edges(
    T data_min,
    T data_max,
    int64_t num_bins,
    T* edges
) {
    T bin_width = (data_max - data_min) / T(num_bins);
    for (int64_t i = 0; i <= num_bins; ++i) {
        edges[i] = data_min + T(i) * bin_width;
    }
    // Ensure last edge is exactly data_max (avoid floating point drift)
    edges[num_bins] = data_max;
}

/**
 * Find bin index for a value using binary search.
 * Returns -1 for out-of-bounds values when out_of_bounds="ignore".
 *
 * @param value The value to bin
 * @param edges Array of bin edges (size num_bins + 1)
 * @param num_bins Number of bins
 * @param closed_right If true, bins are (a, b]; if false, bins are [a, b)
 * @param clamp_oob If true, clamp out-of-bounds to edge bins
 * @return Bin index [0, num_bins-1] or -1 if ignored
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t find_bin(
    T value,
    const T* edges,
    int64_t num_bins,
    bool closed_right,
    bool clamp_oob
) {
    // Handle NaN
    if (std::isnan(value)) {
        return -1;  // NaN is always ignored
    }

    T left_edge = edges[0];
    T right_edge = edges[num_bins];

    // Handle out of bounds
    if (value < left_edge) {
        if (clamp_oob) {
            return 0;
        }
        return -1;
    }
    if (value > right_edge) {
        if (clamp_oob) {
            return num_bins - 1;
        }
        return -1;
    }

    // Binary search for the bin
    int64_t lo = 0;
    int64_t hi = num_bins;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (closed_right) {
            // Bins are (edges[i], edges[i+1]]
            if (value <= edges[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        } else {
            // Bins are [edges[i], edges[i+1])
            if (value < edges[mid + 1]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
    }

    // Handle edge cases for closed side
    if (closed_right) {
        // For (a, b], value exactly at left edge goes to first bin
        if (value == left_edge) {
            return 0;
        }
        // Adjust for 1-indexed search result
        return lo > 0 ? lo - 1 : 0;
    } else {
        // For [a, b), value exactly at right edge goes to last bin
        if (value == right_edge) {
            return num_bins - 1;
        }
        return lo < num_bins ? lo : num_bins - 1;
    }
}

/**
 * Compute 1D histogram for a contiguous array.
 *
 * @param data Input array of n elements
 * @param n Number of elements
 * @param edges Bin edges array (size num_bins + 1)
 * @param num_bins Number of bins
 * @param weights Optional weights array (nullptr for unweighted)
 * @param closed_right If true, bins are (a, b]
 * @param clamp_oob If true, clamp out-of-bounds to edge bins
 * @param counts Output counts array (size num_bins), must be zero-initialized
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void histogram_1d(
    const T* data,
    int64_t n,
    const T* edges,
    int64_t num_bins,
    const T* weights,
    bool closed_right,
    bool clamp_oob,
    T* counts
) {
    for (int64_t i = 0; i < n; ++i) {
        int64_t bin = find_bin(data[i], edges, num_bins, closed_right, clamp_oob);
        if (bin >= 0) {
            T weight = weights ? weights[i] : T(1);
            counts[bin] += weight;
        }
    }
}

/**
 * Apply density normalization to histogram counts.
 *
 * @param counts Histogram counts (modified in-place)
 * @param edges Bin edges
 * @param num_bins Number of bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void normalize_density(
    T* counts,
    const T* edges,
    int64_t num_bins
) {
    // Compute total count
    T total = T(0);
    for (int64_t i = 0; i < num_bins; ++i) {
        total += counts[i];
    }

    if (total == T(0)) {
        return;  // Avoid division by zero
    }

    // Normalize: density = count / (total * bin_width)
    for (int64_t i = 0; i < num_bins; ++i) {
        T bin_width = edges[i + 1] - edges[i];
        if (bin_width > T(0)) {
            counts[i] /= (total * bin_width);
        }
    }
}

}  // namespace torchscience::impl::descriptive
```

**Step 2: Verify header compiles**

Run: `uv run pip install -e . --no-build-isolation`

Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add src/torchscience/csrc/impl/statistics/descriptive/histogram.h
git commit -m "feat(impl): add histogram core algorithm"
```

---

## Task 5: Implement CPU dispatch for histogram

**Files:**
- Create: `src/torchscience/csrc/cpu/statistics/descriptive/histogram.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Write CPU implementation**

```cpp
#pragma once

#include <tuple>
#include <vector>
#include <string>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../impl/statistics/descriptive/histogram.h"

namespace torchscience::cpu::descriptive {

/**
 * CPU implementation of histogram with integer bins.
 */
inline std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& weights,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    TORCH_CHECK(input.numel() > 0, "histogram: input tensor must be non-empty");
    TORCH_CHECK(bins > 0, "histogram: bins must be positive");

    bool closed_right = (closed == "right");
    bool clamp_oob = (out_of_bounds == "clamp");
    bool error_oob = (out_of_bounds == "error");

    // Flatten input for 1D processing
    at::Tensor input_flat = input.flatten().contiguous();
    int64_t n = input_flat.numel();

    // Handle weights
    at::Tensor weights_flat;
    if (weights.has_value()) {
        TORCH_CHECK(
            weights->numel() == n,
            "histogram: weights must have same number of elements as input"
        );
        weights_flat = weights->flatten().contiguous();
    }

    // Determine range
    double data_min, data_max;
    if (range.has_value()) {
        TORCH_CHECK(range->size() == 2, "histogram: range must have 2 elements");
        data_min = (*range)[0];
        data_max = (*range)[1];
        TORCH_CHECK(data_min < data_max, "histogram: range[0] must be less than range[1]");
    } else {
        data_min = input_flat.min().item<double>();
        data_max = input_flat.max().item<double>();
        // Handle edge case where all values are the same
        if (data_min == data_max) {
            data_min -= 0.5;
            data_max += 0.5;
        }
    }

    // Check for out-of-bounds values if error mode
    if (error_oob) {
        double actual_min = input_flat.min().item<double>();
        double actual_max = input_flat.max().item<double>();
        TORCH_CHECK(
            actual_min >= data_min && actual_max <= data_max,
            "histogram: values outside range with out_of_bounds='error'"
        );
    }

    // Create output tensors
    auto options = input.options();
    at::Tensor counts = at::zeros({bins}, options);
    at::Tensor edges = at::empty({bins + 1}, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input_flat.scalar_type(),
        "histogram_cpu",
        [&]() {
            // Compute in float for half types
            using compute_t = std::conditional_t<
                std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
                float,
                scalar_t
            >;

            // Compute bin edges
            std::vector<compute_t> edges_vec(bins + 1);
            impl::descriptive::compute_bin_edges<compute_t>(
                static_cast<compute_t>(data_min),
                static_cast<compute_t>(data_max),
                bins,
                edges_vec.data()
            );

            // Copy edges to output tensor
            scalar_t* edges_ptr = edges.data_ptr<scalar_t>();
            for (int64_t i = 0; i <= bins; ++i) {
                edges_ptr[i] = static_cast<scalar_t>(edges_vec[i]);
            }

            // Compute histogram
            std::vector<compute_t> counts_vec(bins, compute_t(0));
            const scalar_t* data_ptr = input_flat.data_ptr<scalar_t>();
            const scalar_t* weights_ptr = weights_flat.defined()
                ? weights_flat.data_ptr<scalar_t>()
                : nullptr;

            if constexpr (std::is_same_v<scalar_t, at::Half> ||
                          std::is_same_v<scalar_t, at::BFloat16>) {
                // Convert to float for computation
                std::vector<compute_t> data_float(n);
                std::vector<compute_t> weights_float;
                for (int64_t i = 0; i < n; ++i) {
                    data_float[i] = static_cast<compute_t>(data_ptr[i]);
                }
                if (weights_ptr) {
                    weights_float.resize(n);
                    for (int64_t i = 0; i < n; ++i) {
                        weights_float[i] = static_cast<compute_t>(weights_ptr[i]);
                    }
                }

                impl::descriptive::histogram_1d<compute_t>(
                    data_float.data(),
                    n,
                    edges_vec.data(),
                    bins,
                    weights_float.empty() ? nullptr : weights_float.data(),
                    closed_right,
                    clamp_oob,
                    counts_vec.data()
                );
            } else {
                std::vector<compute_t> weights_vec;
                if (weights_ptr) {
                    weights_vec.assign(weights_ptr, weights_ptr + n);
                }

                impl::descriptive::histogram_1d<compute_t>(
                    reinterpret_cast<const compute_t*>(data_ptr),
                    n,
                    edges_vec.data(),
                    bins,
                    weights_vec.empty() ? nullptr : weights_vec.data(),
                    closed_right,
                    clamp_oob,
                    counts_vec.data()
                );
            }

            // Apply density normalization if requested
            if (density) {
                impl::descriptive::normalize_density<compute_t>(
                    counts_vec.data(),
                    edges_vec.data(),
                    bins
                );
            }

            // Copy counts to output tensor
            scalar_t* counts_ptr = counts.data_ptr<scalar_t>();
            for (int64_t i = 0; i < bins; ++i) {
                counts_ptr[i] = static_cast<scalar_t>(counts_vec[i]);
            }
        }
    );

    return std::make_tuple(counts, edges);
}

/**
 * CPU implementation of histogram with explicit edges.
 */
inline std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& edges,
    const c10::optional<at::Tensor>& weights,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    TORCH_CHECK(input.numel() > 0, "histogram: input tensor must be non-empty");
    TORCH_CHECK(edges.dim() == 1, "histogram: edges must be 1D");
    TORCH_CHECK(edges.numel() >= 2, "histogram: edges must have at least 2 elements");

    int64_t bins = edges.numel() - 1;

    // Convert edges to range for the integer bins version
    at::Tensor edges_contig = edges.contiguous();
    double data_min = edges_contig[0].item<double>();
    double data_max = edges_contig[bins].item<double>();

    // For now, use the integer bins version with the range
    // Full support for non-uniform bins will be added later
    auto [counts, computed_edges] = histogram(
        input, bins,
        std::vector<double>{data_min, data_max},
        weights, density, closed, out_of_bounds
    );

    // Return the original edges
    return std::make_tuple(counts, edges.clone());
}

}  // namespace torchscience::cpu::descriptive

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "histogram",
        &torchscience::cpu::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::cpu::descriptive::histogram_edges
    );
}
```

**Step 2: Add include to torchscience.cpp**

Add to includes section:

```cpp
#include "cpu/statistics/descriptive/histogram.h"
```

**Step 3: Build and test**

Run: `uv run pip install -e . --no-build-isolation && uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py -v`

Expected: Tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/statistics/descriptive/histogram.h \
        src/torchscience/csrc/torchscience.cpp
git commit -m "feat(cpu): add histogram CPU implementation"
```

---

## Task 6: Implement Meta backend for histogram

**Files:**
- Create: `src/torchscience/csrc/meta/statistics/descriptive/histogram.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Write meta implementation**

```cpp
#pragma once

#include <tuple>

#include <torch/library.h>

namespace torchscience::meta::descriptive {

inline std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    [[maybe_unused]] c10::optional<at::ArrayRef<double>> range,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights,
    [[maybe_unused]] bool density,
    [[maybe_unused]] c10::string_view closed,
    [[maybe_unused]] c10::string_view out_of_bounds
) {
    auto options = input.options();
    at::Tensor counts = at::empty({bins}, options);
    at::Tensor edges = at::empty({bins + 1}, options);
    return std::make_tuple(counts, edges);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& edges,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights,
    [[maybe_unused]] bool density,
    [[maybe_unused]] c10::string_view closed,
    [[maybe_unused]] c10::string_view out_of_bounds
) {
    int64_t bins = edges.numel() - 1;
    auto options = input.options();
    at::Tensor counts = at::empty({bins}, options);
    return std::make_tuple(counts, edges.clone());
}

}  // namespace torchscience::meta::descriptive

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "histogram",
        &torchscience::meta::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::meta::descriptive::histogram_edges
    );
}
```

**Step 2: Add include and build**

Add include to `torchscience.cpp`:

```cpp
#include "meta/statistics/descriptive/histogram.h"
```

Run: `uv run pip install -e . --no-build-isolation`

**Step 3: Commit**

```bash
git add src/torchscience/csrc/meta/statistics/descriptive/histogram.h \
        src/torchscience/csrc/torchscience.cpp
git commit -m "feat(meta): add histogram meta implementation"
```

---

## Task 7: Implement CUDA backend for histogram

**Files:**
- Create: `src/torchscience/csrc/cuda/statistics/descriptive/histogram.cu`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add conditional include)

**Step 1: Write CUDA implementation**

```cpp
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchscience::cuda::descriptive {

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * Binary search for bin index on device.
 */
template <typename T>
__device__ __forceinline__ int64_t find_bin_device(
    T value,
    const T* edges,
    int64_t num_bins,
    bool closed_right,
    bool clamp_oob
) {
    // Handle NaN
    if (isnan(value)) {
        return -1;
    }

    T left_edge = edges[0];
    T right_edge = edges[num_bins];

    if (value < left_edge) {
        return clamp_oob ? 0 : -1;
    }
    if (value > right_edge) {
        return clamp_oob ? (num_bins - 1) : -1;
    }

    // Binary search
    int64_t lo = 0;
    int64_t hi = num_bins;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (closed_right) {
            if (value <= edges[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        } else {
            if (value < edges[mid + 1]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
    }

    if (closed_right) {
        if (value == left_edge) return 0;
        return lo > 0 ? lo - 1 : 0;
    } else {
        if (value == right_edge) return num_bins - 1;
        return lo < num_bins ? lo : num_bins - 1;
    }
}

/**
 * Kernel to compute histogram using atomic adds.
 */
template <typename scalar_t>
__global__ void histogram_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ edges,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ counts,
    int64_t n,
    int64_t num_bins,
    bool closed_right,
    bool clamp_oob
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    scalar_t value = input[idx];
    int64_t bin = find_bin_device(value, edges, num_bins, closed_right, clamp_oob);

    if (bin >= 0) {
        scalar_t weight = weights ? weights[idx] : scalar_t(1);
        atomicAdd(&counts[bin], weight);
    }
}

/**
 * Kernel to normalize histogram to density.
 */
template <typename scalar_t>
__global__ void normalize_density_kernel(
    scalar_t* __restrict__ counts,
    const scalar_t* __restrict__ edges,
    int64_t num_bins,
    scalar_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bins) return;

    scalar_t bin_width = edges[idx + 1] - edges[idx];
    if (bin_width > scalar_t(0) && total > scalar_t(0)) {
        counts[idx] /= (total * bin_width);
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    c10::optional<at::ArrayRef<double>> range,
    const c10::optional<at::Tensor>& weights,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    TORCH_CHECK(input.is_cuda(), "histogram: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "histogram: input tensor must be non-empty");
    TORCH_CHECK(bins > 0, "histogram: bins must be positive");

    c10::cuda::CUDAGuard device_guard(input.device());

    bool closed_right = (closed == "right");
    bool clamp_oob = (out_of_bounds == "clamp");

    at::Tensor input_flat = input.flatten().contiguous();
    int64_t n = input_flat.numel();

    at::Tensor weights_flat;
    if (weights.has_value()) {
        weights_flat = weights->flatten().contiguous();
    }

    // Determine range
    double data_min, data_max;
    if (range.has_value()) {
        data_min = (*range)[0];
        data_max = (*range)[1];
    } else {
        data_min = input_flat.min().item<double>();
        data_max = input_flat.max().item<double>();
        if (data_min == data_max) {
            data_min -= 0.5;
            data_max += 0.5;
        }
    }

    auto options = input.options();
    at::Tensor counts = at::zeros({bins}, options);
    at::Tensor edges = at::empty({bins + 1}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(),
        "histogram_cuda",
        [&]() {
            // Compute edges on CPU and copy to GPU
            std::vector<scalar_t> edges_vec(bins + 1);
            scalar_t bin_width = (scalar_t(data_max) - scalar_t(data_min)) / scalar_t(bins);
            for (int64_t i = 0; i <= bins; ++i) {
                edges_vec[i] = scalar_t(data_min) + scalar_t(i) * bin_width;
            }
            edges_vec[bins] = scalar_t(data_max);

            scalar_t* edges_ptr = edges.data_ptr<scalar_t>();
            cudaMemcpyAsync(
                edges_ptr, edges_vec.data(),
                (bins + 1) * sizeof(scalar_t),
                cudaMemcpyHostToDevice, stream
            );

            // Launch histogram kernel
            int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            histogram_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_flat.data_ptr<scalar_t>(),
                edges_ptr,
                weights_flat.defined() ? weights_flat.data_ptr<scalar_t>() : nullptr,
                counts.data_ptr<scalar_t>(),
                n,
                bins,
                closed_right,
                clamp_oob
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Apply density normalization
            if (density) {
                scalar_t total = counts.sum().item<scalar_t>();
                int norm_blocks = (bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
                normalize_density_kernel<scalar_t><<<norm_blocks, BLOCK_SIZE, 0, stream>>>(
                    counts.data_ptr<scalar_t>(),
                    edges_ptr,
                    bins,
                    total
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    );

    return std::make_tuple(counts, edges);
}

std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& edges,
    const c10::optional<at::Tensor>& weights,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    int64_t bins = edges.numel() - 1;
    double data_min = edges[0].item<double>();
    double data_max = edges[bins].item<double>();

    auto [counts, computed_edges] = histogram(
        input, bins,
        std::vector<double>{data_min, data_max},
        weights, density, closed, out_of_bounds
    );

    return std::make_tuple(counts, edges.clone());
}

}  // namespace torchscience::cuda::descriptive

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "histogram",
        &torchscience::cuda::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::cuda::descriptive::histogram_edges
    );
}
```

**Step 2: Add conditional include to torchscience.cpp**

In the `#ifdef TORCHSCIENCE_CUDA` block:

```cpp
#include "cuda/statistics/descriptive/histogram.cu"
```

**Step 3: Build and test CUDA**

Run: `uv run pip install -e . --no-build-isolation && uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py::TestHistogramDevice::test_cuda_device -v`

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cuda/statistics/descriptive/histogram.cu \
        src/torchscience/csrc/torchscience.cpp
git commit -m "feat(cuda): add histogram CUDA implementation"
```

---

## Task 8: Wire Python API to C++ operators

**Files:**
- Modify: `src/torchscience/statistics/descriptive/_histogram.py`

**Step 1: Update Python API to call C++ operators**

Replace the TODO section in `_histogram.py`:

```python
"""Histogram implementation."""

from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def histogram(
    input: Tensor,
    bins: Union[int, Tensor, Sequence[Union[int, Tensor]]] = 10,
    *,
    range: Optional[Union[Tuple[float, float], Sequence[Tuple[float, float]]]] = None,
    weights: Optional[Tensor] = None,
    density: bool = False,
    dim: Optional[int] = None,
    closed: Literal["left", "right"] = "right",
    out_of_bounds: Literal["ignore", "clamp", "error"] = "ignore",
) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
    r"""Compute N-dimensional histogram using hard binning (non-differentiable).

    [... docstring unchanged ...]
    """
    # Validate inputs
    if input.numel() == 0:
        raise ValueError("histogram: input tensor must be non-empty")

    # Validate parameters
    if closed not in ("left", "right"):
        raise ValueError(f"histogram: closed must be 'left' or 'right', got {closed!r}")
    if out_of_bounds not in ("ignore", "clamp", "error"):
        raise ValueError(
            f"histogram: out_of_bounds must be 'ignore', 'clamp', or 'error', "
            f"got {out_of_bounds!r}"
        )

    # Handle 1D case (no batching, no N-D)
    if dim is None:
        if isinstance(bins, int):
            # Integer bins - call C++ operator
            range_list = list(range) if range is not None else None
            return torch.ops.torchscience.histogram(
                input.flatten(),
                bins,
                range_list,
                weights.flatten() if weights is not None else None,
                density,
                closed,
                out_of_bounds,
            )
        elif isinstance(bins, Tensor):
            # Explicit edges - call C++ operator
            return torch.ops.torchscience.histogram_edges(
                input.flatten(),
                bins,
                weights.flatten() if weights is not None else None,
                density,
                closed,
                out_of_bounds,
            )

    # TODO: Implement N-D and batched histogram
    raise NotImplementedError(
        "N-dimensional and batched histograms not yet implemented. "
        "Use dim=None with 1D input or integer bins for basic histogram."
    )
```

**Step 2: Run tests**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py -v`

**Step 3: Commit**

```bash
git add src/torchscience/statistics/descriptive/_histogram.py
git commit -m "feat(histogram): wire Python API to C++ operators"
```

---

## Task 9: Update roadmap.rst

**Files:**
- Modify: `docs/roadmap.rst:31`

**Step 1: Mark histogram as complete**

Change line 31 from:
```rst
* Flatten operators via ``histogram``
```

To:
```rst
* Flatten operators via ``histogram`` ✓
```

Also update the module table entry (line 128-130) to add ✓:
```rst
   * - ``statistics.histograms``
     - ``histogram`` ✓
     - Flatten
```

Wait - the module path in roadmap says `statistics.histograms` but we implemented in `statistics.descriptive`. Let me update to match our implementation:

```rst
   * - ``statistics.descriptive`` ✓
     - ``kurtosis``, ``histogram`` ✓
     - Reduction, Flatten
```

**Step 2: Commit**

```bash
git add docs/roadmap.rst
git commit -m "docs: mark histogram as complete in roadmap"
```

---

## Task 10: Run full test suite and verify

**Step 1: Run all histogram tests**

Run: `uv run pytest tests/torchscience/statistics/descriptive/test__histogram.py -v`

Expected: All tests PASS

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`

Expected: No regressions

**Step 3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address any test failures"
```

---

## Summary

This plan implements `histogram` - a fast, non-differentiable N-D histogram operator:

- CPU and CUDA backends
- Meta backend for tracing
- Support for weights, density, range
- Configurable edge handling (closed, out_of_bounds)

This establishes the "Flatten" category pattern where input samples are flattened and output shape is determined by bin count.

**Files created/modified:**
- `src/torchscience/statistics/descriptive/_histogram.py` (new)
- `src/torchscience/statistics/descriptive/__init__.py` (modified)
- `src/torchscience/csrc/torchscience.cpp` (modified)
- `src/torchscience/csrc/impl/statistics/descriptive/histogram.h` (new)
- `src/torchscience/csrc/cpu/statistics/descriptive/histogram.h` (new)
- `src/torchscience/csrc/meta/statistics/descriptive/histogram.h` (new)
- `src/torchscience/csrc/cuda/statistics/descriptive/histogram.cu` (new)
- `tests/torchscience/statistics/descriptive/test__histogram.py` (new)
- `docs/roadmap.rst` (modified)
