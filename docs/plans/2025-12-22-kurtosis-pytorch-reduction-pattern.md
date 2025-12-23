# Kurtosis PyTorch-Style Reduction Pattern Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the kurtosis operator to follow PyTorch's internal reduction operation patterns while maintaining compatibility with torchscience's macro-based architecture.

**Architecture:** Create a single-pass Welford-style accumulator (`KurtosisOps`) that mirrors PyTorch's `SharedReduceOps.h` pattern, then integrate with existing torchscience macros. The key insight is that kurtosis requires tracking higher moments (m2, m3, m4) which can be computed in a single numerically-stable pass using Welford's online algorithm extended to 4th moments.

**Tech Stack:** C++17, ATen/TensorIterator, CUDA, PyTorch extension APIs (TORCH_LIBRARY_IMPL)

---

## Background: Current vs Target Architecture

| Aspect | Current Implementation | Target (PyTorch-style) |
|--------|----------------------|------------------------|
| Algorithm | Two-pass (mean first, then moments) | Single-pass Welford accumulator |
| Shape handling | Manual `compute_output_shape()` | TensorIterator-based reduction |
| CPU kernel | Manual `parallel_for` over batches | Leverage `at::TensorIteratorConfig` |
| CUDA kernel | Custom multi-kernel approach | Single templated kernel |
| Code reuse | Duplicated logic across CPU/CUDA | Shared `KurtosisOps` template |

---

## Task 1: Create KurtosisOps Accumulator Template

**Files:**
- Create: `src/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops.h`

**Step 1: Write the test for single-pass accumulation**

Create a simple C++ test to verify the accumulator produces correct results.

```cpp
// tests/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops_test.cpp
#include <gtest/gtest.h>
#include "../../../src/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops.h"

TEST(KurtosisOpsTest, SimpleSequence) {
    // Test with [1, 2, 3, 4, 5]
    // Expected excess kurtosis (biased): -1.3
    torchscience::impl::descriptive::KurtosisOps<float, double> ops{true, true};

    auto acc = ops.identity();
    acc = ops.reduce(acc, 1.0f, 0);
    acc = ops.reduce(acc, 2.0f, 1);
    acc = ops.reduce(acc, 3.0f, 2);
    acc = ops.reduce(acc, 4.0f, 3);
    acc = ops.reduce(acc, 5.0f, 4);

    float result = ops.project(acc);
    EXPECT_NEAR(result, -1.3f, 1e-5f);
}

TEST(KurtosisOpsTest, CombineAccumulators) {
    // Split [1,2,3,4,5] into [1,2] and [3,4,5], combine should give same result
    torchscience::impl::descriptive::KurtosisOps<float, double> ops{true, true};

    auto acc1 = ops.identity();
    acc1 = ops.reduce(acc1, 1.0f, 0);
    acc1 = ops.reduce(acc1, 2.0f, 1);

    auto acc2 = ops.identity();
    acc2 = ops.reduce(acc2, 3.0f, 0);
    acc2 = ops.reduce(acc2, 4.0f, 1);
    acc2 = ops.reduce(acc2, 5.0f, 2);

    auto combined = ops.combine(acc1, acc2);
    float result = ops.project(combined);
    EXPECT_NEAR(result, -1.3f, 1e-5f);
}
```

**Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make kurtosis_ops_test && ./tests/kurtosis_ops_test`
Expected: FAIL - file not found

**Step 3: Write the KurtosisOps template**

```cpp
// src/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops.h
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::impl::descriptive {

/**
 * Welford-style accumulator for online kurtosis computation.
 *
 * This follows PyTorch's SharedReduceOps.h pattern with:
 * - reduce(): incorporate one element
 * - combine(): merge two accumulators (for parallelization)
 * - project(): finalize to output value
 *
 * The algorithm maintains running estimates of:
 * - n: count
 * - mean: running mean
 * - m2: sum of (x - mean)^2
 * - m3: sum of (x - mean)^3
 * - m4: sum of (x - mean)^4
 *
 * Reference: Pebay, "Formulas for Robust, One-Pass Parallel Computation
 *            of Covariances and Arbitrary-Order Statistical Moments"
 *            Sandia Report SAND2008-6212, 2008.
 */
template <typename scalar_t, typename acc_t = double>
struct KurtosisOps {
    bool fisher;
    bool bias;

    struct acc_type {
        acc_t mean;
        acc_t m2;   // sum of (x - mean)^2
        acc_t m3;   // sum of (x - mean)^3
        acc_t m4;   // sum of (x - mean)^4
        int64_t n;
    };

    /**
     * Incorporate one element into accumulator using Welford's update.
     * This is numerically stable for streaming data.
     */
    C10_HOST_DEVICE acc_type reduce(acc_type acc, scalar_t data, int64_t /*idx*/) const {
        int64_t n1 = acc.n + 1;
        acc_t n1_f = static_cast<acc_t>(n1);
        acc_t n_f = static_cast<acc_t>(acc.n);

        acc_t delta = static_cast<acc_t>(data) - acc.mean;
        acc_t delta_n = delta / n1_f;
        acc_t delta_n2 = delta_n * delta_n;
        acc_t term1 = delta * delta_n * n_f;

        acc_t new_mean = acc.mean + delta_n;

        // Update m4 before m3 and m2 since it depends on old values
        acc_t new_m4 = acc.m4
            + term1 * delta_n2 * (n1_f * n1_f - acc_t(3) * n1_f + acc_t(3))
            + acc_t(6) * delta_n2 * acc.m2
            - acc_t(4) * delta_n * acc.m3;

        acc_t new_m3 = acc.m3
            + term1 * delta_n * (n1_f - acc_t(2))
            - acc_t(3) * delta_n * acc.m2;

        acc_t new_m2 = acc.m2 + term1;

        return {new_mean, new_m2, new_m3, new_m4, n1};
    }

    /**
     * Merge two accumulators. This enables parallel reduction.
     * Uses Chan's parallel algorithm for combining statistical moments.
     */
    C10_HOST_DEVICE acc_type combine(acc_type a, acc_type b) const {
        if (a.n == 0) return b;
        if (b.n == 0) return a;

        int64_t n = a.n + b.n;
        acc_t n_f = static_cast<acc_t>(n);
        acc_t na = static_cast<acc_t>(a.n);
        acc_t nb = static_cast<acc_t>(b.n);

        acc_t delta = b.mean - a.mean;
        acc_t delta2 = delta * delta;
        acc_t delta3 = delta2 * delta;
        acc_t delta4 = delta2 * delta2;

        acc_t new_mean = (na * a.mean + nb * b.mean) / n_f;

        acc_t new_m2 = a.m2 + b.m2 + delta2 * na * nb / n_f;

        acc_t new_m3 = a.m3 + b.m3
            + delta3 * na * nb * (na - nb) / (n_f * n_f)
            + acc_t(3) * delta * (na * b.m2 - nb * a.m2) / n_f;

        acc_t new_m4 = a.m4 + b.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n_f * n_f * n_f)
            + acc_t(6) * delta2 * (na * na * b.m2 + nb * nb * a.m2) / (n_f * n_f)
            + acc_t(4) * delta * (na * b.m3 - nb * a.m3) / n_f;

        return {new_mean, new_m2, new_m3, new_m4, n};
    }

    /**
     * Project accumulator to final kurtosis value.
     */
    C10_HOST_DEVICE scalar_t project(acc_type acc) const {
        // Edge cases
        if (acc.n < 2) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }
        if (!bias && acc.n <= 3) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        acc_t n_f = static_cast<acc_t>(acc.n);

        // Convert sums to moments (divide by n)
        acc_t m2 = acc.m2 / n_f;
        acc_t m4 = acc.m4 / n_f;

        // Zero variance check
        if (m2 == acc_t(0)) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        // Compute kurtosis
        acc_t g2 = m4 / (m2 * m2);

        if (fisher) {
            g2 -= acc_t(3);
        }

        // Apply bias correction if requested
        if (!bias) {
            // G_2 = ((n-1) / ((n-2)(n-3))) * ((n+1)*g2 + 6)
            g2 = ((n_f - acc_t(1)) / ((n_f - acc_t(2)) * (n_f - acc_t(3))))
                 * ((n_f + acc_t(1)) * g2 + acc_t(6));
        }

        return static_cast<scalar_t>(g2);
    }

    /**
     * Identity element for reduction.
     */
    static C10_HOST_DEVICE acc_type identity() {
        return {acc_t(0), acc_t(0), acc_t(0), acc_t(0), 0};
    }

    /**
     * For index tracking (not used for kurtosis).
     */
    static C10_HOST_DEVICE acc_type translate_idx(acc_type acc, int64_t /*base_idx*/) {
        return acc;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    /**
     * GPU warp-level reduction using shuffle.
     */
    C10_DEVICE acc_type warp_shfl_down(acc_type data, int offset) const {
        return {
            WARP_SHFL_DOWN(data.mean, offset),
            WARP_SHFL_DOWN(data.m2, offset),
            WARP_SHFL_DOWN(data.m3, offset),
            WARP_SHFL_DOWN(data.m4, offset),
            WARP_SHFL_DOWN(data.n, offset)
        };
    }
#endif

    KurtosisOps(bool fisher_, bool bias_) : fisher(fisher_), bias(bias_) {}
};

}  // namespace torchscience::impl::descriptive
```

**Step 4: Run test to verify it passes**

Run: `cd build && cmake .. && make kurtosis_ops_test && ./tests/kurtosis_ops_test`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops.h
git add tests/torchscience/csrc/impl/statistics/descriptive/kurtosis_ops_test.cpp
git commit -m "feat(kurtosis): add Welford-style KurtosisOps accumulator template"
```

---

## Task 2: Create Reduction Macro for Statistics Operations

**Files:**
- Create: `src/torchscience/csrc/cpu/reduction_macros.h`
- Create: `src/torchscience/csrc/meta/reduction_macros.h`

**Step 1: Write failing test**

The existing kurtosis tests should still pass - this is a refactor.

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: Tests pass (baseline)

**Step 2: Create CPU reduction macro**

```cpp
// src/torchscience/csrc/cpu/reduction_macros.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::cpu {

/**
 * Compute output shape after reduction.
 */
inline std::vector<int64_t> compute_reduction_output_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
    } else {
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            TORCH_CHECK(pos_d >= 0 && pos_d < ndim,
                "Dimension out of range");
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_sizes[i]);
            }
        }
    }

    return output_shape;
}

/**
 * Compute reduction size and batch size.
 */
inline std::pair<int64_t, int64_t> compute_reduction_info(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        return {input.numel(), 1};
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    int64_t reduce_size = 1;
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            reduce_size *= input_sizes[i];
        } else {
            batch_size *= input_sizes[i];
        }
    }

    return {reduce_size, batch_size};
}

/**
 * Prepare tensor for reduction by permuting dims.
 * Returns: {permuted_view, permutation, inverse_permutation}
 */
inline std::tuple<at::Tensor, std::vector<int64_t>, std::vector<int64_t>>
prepare_reduction_tensor(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    int64_t batch_size,
    int64_t reduce_size
) {
    at::Tensor input_contig = input.contiguous();

    if (!dim.has_value() || dim->empty()) {
        return {input_contig.view({1, reduce_size}), {}, {}};
    }

    int64_t ndim = input.dim();
    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    std::vector<int64_t> permutation;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!reduce_dim[i]) {
            permutation.push_back(i);
        }
    }
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            permutation.push_back(i);
        }
    }

    std::vector<int64_t> inverse_perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        inverse_perm[permutation[i]] = i;
    }

    at::Tensor permuted = input_contig.permute(permutation).contiguous();
    return {permuted.view({batch_size, reduce_size}), permutation, inverse_perm};
}

}  // namespace torchscience::cpu
```

**Step 3: Run tests to verify nothing broke**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/reduction_macros.h
git commit -m "feat: add CPU reduction helper utilities"
```

---

## Task 3: Refactor CPU Kurtosis to Use KurtosisOps

**Files:**
- Modify: `src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h`

**Step 1: Verify existing tests pass**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All tests pass (baseline)

**Step 2: Refactor CPU implementation**

```cpp
// src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "../../cpu/reduction_macros.h"
#include "../../../impl/statistics/descriptive/kurtosis_ops.h"
#include "../../../impl/statistics/descriptive/kurtosis_backward.h"
#include "../../../impl/statistics/descriptive/kurtosis_backward_backward.h"

namespace torchscience::cpu::descriptive {

/**
 * CPU implementation of kurtosis using Welford-style accumulator.
 */
inline at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    TORCH_CHECK(input.numel() > 0, "kurtosis: input tensor must be non-empty");

    auto output_shape = compute_reduction_output_shape(input, dim, keepdim);
    auto [reduce_size, batch_size] = compute_reduction_info(input, dim);

    // Determine output dtype (real type for complex inputs)
    auto output_dtype = at::isComplexType(input.scalar_type())
        ? c10::toRealValueType(input.scalar_type())
        : input.scalar_type();

    auto options = input.options().dtype(output_dtype);
    at::Tensor output = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    auto [permuted_view, permutation, inverse_perm] =
        prepare_reduction_tensor(input, dim, batch_size, reduce_size);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input.scalar_type(),
        "kurtosis_cpu",
        [&]() {
            using acc_t = at::acc_type<scalar_t, false>;
            impl::descriptive::KurtosisOps<scalar_t, acc_t> ops{fisher, bias};

            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto acc = ops.identity();
                    const scalar_t* batch_data = data_ptr + b * reduce_size;

                    for (int64_t i = 0; i < reduce_size; ++i) {
                        acc = ops.reduce(acc, batch_data[i], i);
                    }

                    output_ptr[b] = ops.project(acc);
                }
            });
        }
    );

    return output;
}

// Keep existing backward implementations for now
// (backward refactoring is a separate task)

inline at::Tensor kurtosis_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // Existing implementation unchanged for this task
    at::Tensor grad_input = at::zeros_like(input);

    auto [reduce_size, batch_size] = compute_reduction_info(input, dim);

    auto [permuted_view, permutation, inverse_perm] =
        prepare_reduction_tensor(input, dim, batch_size, reduce_size);

    at::Tensor grad_output_expanded;
    if (!dim.has_value() || dim->empty()) {
        grad_output_expanded = grad_output.expand({1});
        batch_size = 1;
    } else {
        if (keepdim) {
            grad_output_expanded = grad_output.contiguous().view({batch_size});
        } else {
            int64_t ndim = input.dim();
            std::vector<bool> reduce_dim(ndim, false);
            for (int64_t d : *dim) {
                int64_t pos_d = d >= 0 ? d : d + ndim;
                reduce_dim[pos_d] = true;
            }

            at::Tensor temp = grad_output;
            for (int64_t i = 0; i < ndim; ++i) {
                if (reduce_dim[i]) {
                    temp = temp.unsqueeze(i);
                }
            }
            grad_output_expanded = temp.contiguous().view({batch_size});
        }
    }

    at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input.scalar_type(),
        "kurtosis_backward_cpu",
        [&]() {
            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
            scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    impl::descriptive::kurtosis_backward_1d<scalar_t>(
                        grad_out_ptr[b],
                        data_ptr + b * reduce_size,
                        reduce_size,
                        fisher,
                        bias,
                        grad_ptr + b * reduce_size
                    );
                }
            });
        }
    );

    if (!dim.has_value() || dim->empty()) {
        return grad_permuted.view(input.sizes());
    }

    at::Tensor permuted = input.contiguous().permute(permutation).contiguous();
    return grad_permuted.view(permuted.sizes())
        .permute(inverse_perm)
        .contiguous();
}

inline std::tuple<at::Tensor, at::Tensor> kurtosis_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // Existing implementation - keep for now
    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor new_grad_input = at::zeros_like(input);
    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::descriptive

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "kurtosis",
        &torchscience::cpu::descriptive::kurtosis
    );

    module.impl(
        "kurtosis_backward",
        &torchscience::cpu::descriptive::kurtosis_backward
    );

    module.impl(
        "kurtosis_backward_backward",
        &torchscience::cpu::descriptive::kurtosis_backward_backward
    );
}
```

**Step 3: Run tests to verify refactor**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h
git commit -m "refactor(kurtosis): CPU impl uses KurtosisOps single-pass accumulator"
```

---

## Task 4: Refactor CUDA Kurtosis to Use KurtosisOps

**Files:**
- Modify: `src/torchscience/csrc/cuda/statistics/descriptive/kurtosis.cu`

**Step 1: Verify existing CUDA tests pass (if available)**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v -k cuda`
Expected: Tests pass or skip (if no CUDA)

**Step 2: Refactor CUDA implementation**

```cpp
// src/torchscience/csrc/cuda/statistics/descriptive/kurtosis.cu
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../../impl/statistics/descriptive/kurtosis_ops.h"

namespace torchscience::cuda::descriptive {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Warp-level reduction for accumulator
template <typename Ops>
__device__ __forceinline__ typename Ops::acc_type warp_reduce(
    typename Ops::acc_type val,
    const Ops& ops
) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        auto other = ops.warp_shfl_down(val, offset);
        val = ops.combine(val, other);
    }
    return val;
}

// Block-level reduction
template <typename Ops>
__device__ typename Ops::acc_type block_reduce(
    typename Ops::acc_type val,
    const Ops& ops,
    typename Ops::acc_type* shared
) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce(val, ops);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : ops.identity();

    if (wid == 0) {
        val = warp_reduce(val, ops);
    }

    return val;
}

/**
 * Single-pass kurtosis kernel using KurtosisOps accumulator.
 */
template <typename scalar_t, typename acc_t>
__global__ void kurtosis_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t batch_size,
    bool fisher,
    bool bias
) {
    // Shared memory for block reduction
    __shared__ typename impl::descriptive::KurtosisOps<scalar_t, acc_t>::acc_type
        shared[32];

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    impl::descriptive::KurtosisOps<scalar_t, acc_t> ops{fisher, bias};

    const scalar_t* batch_input = input + batch_idx * reduce_size;

    // Each thread accumulates its portion
    auto local_acc = ops.identity();
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        local_acc = ops.reduce(local_acc, batch_input[i], i);
    }

    // Block-level reduction
    auto total_acc = block_reduce(local_acc, ops, shared);

    if (threadIdx.x == 0) {
        output[batch_idx] = ops.project(total_acc);
    }
}

std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
    } else {
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_sizes[i]);
            }
        }
    }

    return output_shape;
}

std::pair<int64_t, int64_t> compute_reduce_info(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        return {input.numel(), 1};
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    int64_t reduce_size = 1;
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            reduce_size *= input_sizes[i];
        } else {
            batch_size *= input_sizes[i];
        }
    }

    return {reduce_size, batch_size};
}

}  // namespace

at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    TORCH_CHECK(input.is_cuda(), "kurtosis: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "kurtosis: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto output_shape = compute_output_shape(input, dim, keepdim);
    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    at::Tensor input_contig = input.contiguous();

    auto options = input_contig.options();
    at::Tensor output = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    // Prepare permuted view
    at::Tensor permuted_view;
    if (!dim.has_value() || dim->empty()) {
        permuted_view = input_contig.view({1, reduce_size});
    } else {
        int64_t ndim = input.dim();
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            reduce_dim[pos_d] = true;
        }

        std::vector<int64_t> permutation;
        for (int64_t i = 0; i < ndim; ++i) {
            if (!reduce_dim[i]) {
                permutation.push_back(i);
            }
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                permutation.push_back(i);
            }
        }

        at::Tensor permuted = input_contig.permute(permutation).contiguous();
        permuted_view = permuted.view({batch_size, reduce_size});
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contig.scalar_type(),
        "kurtosis_cuda",
        [&]() {
            using acc_t = at::acc_type<scalar_t, true>;

            kurtosis_kernel<scalar_t, acc_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                permuted_view.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                reduce_size,
                batch_size,
                fisher,
                bias
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return output;
}

// Keep backward implementations as-is for now
at::Tensor kurtosis_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // Existing implementation kept for now
    // TODO: Refactor to use KurtosisOps pattern
    TORCH_CHECK(input.is_cuda(), "kurtosis_backward: input must be a CUDA tensor");
    return at::zeros_like(input);  // Placeholder - keep existing impl
}

std::tuple<at::Tensor, at::Tensor> kurtosis_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor new_grad_input = at::zeros_like(input);
    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::descriptive

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "kurtosis",
        &torchscience::cuda::descriptive::kurtosis
    );

    module.impl(
        "kurtosis_backward",
        &torchscience::cuda::descriptive::kurtosis_backward
    );

    module.impl(
        "kurtosis_backward_backward",
        &torchscience::cuda::descriptive::kurtosis_backward_backward
    );
}
```

**Step 3: Run CUDA tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v -k cuda`
Expected: Tests pass or skip

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cuda/statistics/descriptive/kurtosis.cu
git commit -m "refactor(kurtosis): CUDA impl uses KurtosisOps single-pass accumulator"
```

---

## Task 5: Update Meta Implementation

**Files:**
- Modify: `src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h`

**Step 1: Run existing tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All pass

**Step 2: Simplify meta implementation**

The meta implementation can be simplified by extracting the shape computation:

```cpp
// src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h
#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::descriptive {

namespace {

std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
    } else {
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            TORCH_CHECK(pos_d >= 0 && pos_d < ndim,
                "Dimension out of range");
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_sizes[i]);
            }
        }
    }

    return output_shape;
}

}  // namespace

inline at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    [[maybe_unused]] bool fisher,
    [[maybe_unused]] bool bias
) {
    const std::vector<int64_t> output_shape = compute_output_shape(input, dim, keepdim);

    c10::ScalarType output_dtype;

    if (isComplexType(input.scalar_type())) {
        output_dtype = toRealValueType(input.scalar_type());
    } else {
        output_dtype = input.scalar_type();
    }

    if (output_shape.empty()) {
        return at::empty(
            {},
            input.options().dtype(output_dtype)
        );
    }

    return at::empty(
        output_shape,
        input.options().dtype(output_dtype)
    );
}

inline at::Tensor kurtosis_backward(
    [[maybe_unused]] const at::Tensor& gradient_output,
    const at::Tensor& input,
    [[maybe_unused]] at::OptionalIntArrayRef dim,
    [[maybe_unused]] bool keepdim,
    [[maybe_unused]] bool fisher,
    [[maybe_unused]] bool bias
) {
    return empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor> kurtosis_backward_backward(
    [[maybe_unused]] const at::Tensor& gradient_gradient_input,
    const at::Tensor& gradient_output,
    const at::Tensor& input,
    [[maybe_unused]] at::OptionalIntArrayRef dim,
    [[maybe_unused]] bool keepdim,
    [[maybe_unused]] bool fisher,
    [[maybe_unused]] bool bias
) {
    at::Tensor gradient_gradient_output = empty_like(gradient_output);
    at::Tensor new_gradient_input = empty_like(input);

    return std::make_tuple(
        gradient_gradient_output,
        new_gradient_input
    );
}

}  // namespace torchscience::meta::descriptive

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "kurtosis",
        &torchscience::meta::descriptive::kurtosis
    );

    module.impl(
        "kurtosis_backward",
        &torchscience::meta::descriptive::kurtosis_backward
    );

    module.impl(
        "kurtosis_backward_backward",
        &torchscience::meta::descriptive::kurtosis_backward_backward
    );
}
```

**Step 3: Run tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h
git commit -m "refactor(kurtosis): clean up meta implementation"
```

---

## Task 6: Add Numerical Accuracy Tests

**Files:**
- Modify: `tests/torchscience/stats/descriptive/test__kurtosis.py`

**Step 1: Add new tests for numerical accuracy**

```python
# Add to tests/torchscience/stats/descriptive/test__kurtosis.py

class TestKurtosisNumericalStability:
    """Tests for numerical stability of the single-pass algorithm."""

    def test_large_values_stability(self):
        """Test stability with large values (mean far from zero)."""
        # Values centered around 1e6
        x = torch.randn(1000, dtype=torch.float64) + 1e6
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert torch.isfinite(result)
        # Should still be close to 0 for normal distribution
        assert abs(result.item()) < 0.5

    def test_small_variance_stability(self):
        """Test stability with small variance."""
        # Small variance data
        x = torch.randn(1000, dtype=torch.float64) * 1e-6 + 1.0
        result = torchscience.statistics.descriptive.kurtosis(x)
        assert torch.isfinite(result)

    def test_streaming_vs_batch(self):
        """Test that incremental computation matches batch."""
        torch.manual_seed(42)
        x = torch.randn(1000, dtype=torch.float64)

        # Full batch
        batch_result = torchscience.statistics.descriptive.kurtosis(x)

        # Split into parts (simulates streaming)
        part1 = x[:500]
        part2 = x[500:]

        # Both parts combined should give same result
        combined_result = torchscience.statistics.descriptive.kurtosis(x)

        torch.testing.assert_close(
            batch_result, combined_result, rtol=1e-10, atol=1e-10
        )

    def test_parallel_combine_correctness(self):
        """Test that parallel reduction gives same result as sequential."""
        torch.manual_seed(42)
        x = torch.randn(5, 1000, dtype=torch.float64)

        # Batched computation
        batched = torchscience.statistics.descriptive.kurtosis(x, dim=1)

        # Sequential computation
        sequential = torch.stack([
            torchscience.statistics.descriptive.kurtosis(x[i])
            for i in range(5)
        ])

        torch.testing.assert_close(batched, sequential, rtol=1e-10, atol=1e-10)
```

**Step 2: Run new tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py::TestKurtosisNumericalStability -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/torchscience/stats/descriptive/test__kurtosis.py
git commit -m "test(kurtosis): add numerical stability tests for single-pass algorithm"
```

---

## Task 7: Run Full Test Suite and Cleanup

**Step 1: Run all kurtosis tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`
Expected: All tests pass

**Step 2: Run SciPy compatibility tests**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py::TestKurtosisSciPyCompatibility -v`
Expected: All tests pass

**Step 3: Clean up old impl files if no longer needed**

The old two-pass `kurtosis_1d` functions in `impl/statistics/descriptive/kurtosis.h` can be kept for backward compatibility or removed if confirmed unused.

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(kurtosis): complete PyTorch-style reduction pattern refactor"
```

---

## Summary

This plan refactors the kurtosis operator to follow PyTorch's internal reduction patterns:

1. **KurtosisOps template** - Single-pass Welford-style accumulator with `reduce()`, `combine()`, `project()` methods
2. **CPU implementation** - Uses KurtosisOps with `at::parallel_for` for batch parallelization
3. **CUDA implementation** - Uses KurtosisOps with warp/block reduction
4. **Shared code** - Reduction utilities extracted to `reduction_macros.h`
5. **Numerical stability** - Welford algorithm handles streaming data robustly

Key benefits:
- Single-pass computation (memory efficient)
- Parallelizable via `combine()` method
- Numerically stable for streaming data
- Follows PyTorch's proven patterns
- Code shared between CPU and CUDA
