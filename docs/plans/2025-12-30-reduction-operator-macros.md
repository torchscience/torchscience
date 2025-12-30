# Reduction Operator Macros Implementation Plan

> **Status:** COMPLETED (2025-12-30)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Completion Summary

- **Tasks 1-6:** Core infrastructure (CPU, Meta, Autograd, Autocast macros) and test operator (sum_squares) - COMPLETE
- **Task 7:** Comprehensive test suite (23 tests) - COMPLETE
- **Task 8:** Binary macros - SKIPPED (not needed for kurtosis)
- **Task 9:** Kurtosis macro migration - COMPLETE

## Extra Parameters Fix (2025-12-30)

The original `__VA_ARGS__` design couldn't properly separate parameter declarations from names.
Added `_EX` (extended) macro variants with explicit separation:

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `EXTRA_PARAMS` | Function signatures | `TSCI_EXTRA(bool fisher, bool bias)` |
| `EXTRA_ARGS` | Function calls | `TSCI_EXTRA(fisher, bias)` |
| `EXTRA_TYPES` | Dispatcher `typed<>` | `TSCI_TYPES(bool, bool)` |
| `EXTRA_SAVE` | Save to ctx in forward | `TSCI_SAVE(ctx->saved_data["fisher"] = fisher;)` |
| `EXTRA_LOAD` | Load from ctx in backward | `TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool();)` |
| `EXTRA_GRAD_PLACEHOLDERS` | Backward return placeholders | `TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())` |

**Kurtosis migration:**
- CPU: Uses `_EX` macro (~525 lines → 14 lines)
- Meta: Uses `_EX` macro (~130 lines → 14 lines)
- Autocast: Uses `_EX` macro (~57 lines → 14 lines)
- Autograd: Uses `_EX` macro (~260 lines → 25 lines) - now supports extra params via SAVE/LOAD

**Test results:** All 42 kurtosis tests pass (2 pre-existing complex number failures), all 23 sum_squares tests pass.

---

**Goal:** Create macro infrastructure for reduction operators that parallels the existing pointwise operator macros, reducing boilerplate when adding new reduction operators.

**Architecture:** Two macro families (dim-based reductions with `dim`/`keepdim` params, and fixed reductions with configurable mode). Each family has unary through quaternary variants across four backends (CPU, Meta, Autograd, Autocast). Kernels use pointer+length interface and live in `kernel/` mirroring the module structure.

**Tech Stack:** C++ macros, PyTorch ATen, TORCH_LIBRARY_IMPL, AT_DISPATCH, at::parallel_for

---

## Phase 1: Core Infrastructure

### Task 1: Create CPU Reduction Macros

**Files:**
- Create: `src/torchscience/csrc/cpu/reduction_macros.h`

**Step 1: Create the CPU reduction macros header**

```cpp
// src/torchscience/csrc/cpu/reduction_macros.h
#pragma once

#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

// =============================================================================
// REDUCTION UTILITIES (internal to this header)
// =============================================================================

namespace torchscience::cpu::reduction_detail {

inline std::vector<int64_t> compute_reduction_shape(
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
                "Dimension out of range (expected to be in range of [",
                -ndim, ", ", ndim - 1, "], but got ", d, ")");
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

inline std::pair<int64_t, int64_t> compute_reduction_sizes(
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

inline std::vector<int64_t> build_reduction_permutation(
    int64_t ndim,
    at::OptionalIntArrayRef dim
) {
    if (!dim.has_value() || dim->empty()) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }

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

    return permutation;
}

inline std::vector<int64_t> build_inverse_permutation(
    const std::vector<int64_t>& permutation
) {
    std::vector<int64_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = static_cast<int64_t>(i);
    }
    return inverse;
}

enum class ReductionMode {
    LAST_DIM,
    ALL_DIMS
};

}  // namespace torchscience::cpu::reduction_detail

// =============================================================================
// DIM-BASED REDUCTION MACROS
// =============================================================================

/**
 * CPU macro for unary dim-based reduction operators.
 *
 * Generates forward, backward, and backward_backward functions that:
 * - Handle arbitrary dim/keepdim combinations
 * - Permute input to make reduced dims contiguous
 * - Dispatch over floating and complex types
 * - Use parallel_for over batch dimensions
 * - Register with TORCH_LIBRARY_IMPL
 *
 * Kernel interface (in namespace torchscience::kernel::NS):
 *   template<T> T name(const T* data, int64_t n, EXTRA_ARGS...)
 *   template<T> void name_backward(T grad_out, const T* data, int64_t n, EXTRA_ARGS..., T* grad_input)
 *   template<T> void name_backward_backward(const T* gg_input, T grad_out, const T* data, int64_t n, EXTRA_ARGS..., T& gg_output, T* new_grad_input)
 *
 * @param NS Namespace suffix (e.g., statistics::descriptive)
 * @param name Operator name (e.g., kurtosis)
 * @param arg Tensor argument name (e.g., input)
 * @param ... Extra arguments with types (e.g., bool fisher, bool bias)
 */
#define TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)       \
namespace torchscience::cpu::NS {                                               \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    TORCH_CHECK(arg.numel() > 0, #name ": input tensor must be non-empty");     \
                                                                                \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    auto output_shape = compute_reduction_shape(arg, dim, keepdim);             \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    auto output_dtype = at::isComplexType(arg.scalar_type())                    \
        ? c10::toRealValueType(arg.scalar_type())                               \
        : arg.scalar_type();                                                    \
                                                                                \
    auto options = arg##_contig.options().dtype(output_dtype);                  \
    at::Tensor output = output_shape.empty()                                    \
        ? at::empty({}, options)                                                \
        : at::empty(output_shape, options);                                     \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_all",                                                   \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t result = kernel::NS::name<scalar_t>(                   \
                    data_ptr, arg##_contig.numel()                              \
                    __VA_OPT__(, __VA_ARGS__)                                   \
                );                                                              \
                output.fill_(result);                                           \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg.dim(), dim);             \
    auto permuted = arg##_contig.permute(permutation).contiguous();             \
    auto permuted_view = permuted.view({batch_size, reduce_size});              \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        arg##_contig.scalar_type(),                                             \
        #name "_cpu_dim",                                                       \
        [&]() {                                                                 \
            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();      \
            scalar_t* output_ptr = output.data_ptr<scalar_t>();                 \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    output_ptr[b] = kernel::NS::name<scalar_t>(                 \
                        data_ptr + b * reduce_size,                             \
                        reduce_size                                             \
                        __VA_OPT__(, __VA_ARGS__)                               \
                    );                                                          \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    return output;                                                              \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_input = at::zeros_like(arg);                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_input_contig = grad_input.contiguous();                           \
                                                                                \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_all",                                          \
            [&]() {                                                             \
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* grad_ptr = grad_input_contig.data_ptr<scalar_t>();    \
                                                                                \
                kernel::NS::name##_backward<scalar_t>(                          \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    __VA_OPT__(, __VA_ARGS__),                                  \
                    grad_ptr                                                    \
                );                                                              \
            }                                                                   \
        );                                                                      \
        return grad_input_contig;                                               \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg.dim(), dim);             \
    auto permuted = arg##_contig.permute(permutation).contiguous();             \
    auto permuted_view = permuted.view({batch_size, reduce_size});              \
                                                                                \
    at::Tensor grad_output_expanded;                                            \
    if (keepdim) {                                                              \
        grad_output_expanded = grad_output.contiguous().view({batch_size});     \
    } else {                                                                    \
        int64_t ndim = arg.dim();                                               \
        std::vector<bool> reduce_dim(ndim, false);                              \
        for (int64_t d : *dim) {                                                \
            int64_t pos_d = d >= 0 ? d : d + ndim;                              \
            reduce_dim[pos_d] = true;                                           \
        }                                                                       \
        at::Tensor temp = grad_output;                                          \
        for (int64_t i = 0; i < ndim; ++i) {                                    \
            if (reduce_dim[i]) {                                                \
                temp = temp.unsqueeze(i);                                       \
            }                                                                   \
        }                                                                       \
        grad_output_expanded = temp.contiguous().view({batch_size});            \
    }                                                                           \
                                                                                \
    at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, arg.options());\
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        arg##_contig.scalar_type(),                                             \
        #name "_backward_cpu_dim",                                              \
        [&]() {                                                                 \
            const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();      \
            const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();\
            scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();            \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    kernel::NS::name##_backward<scalar_t>(                      \
                        grad_out_ptr[b],                                        \
                        data_ptr + b * reduce_size,                             \
                        reduce_size                                             \
                        __VA_OPT__(, __VA_ARGS__),                              \
                        grad_ptr + b * reduce_size                              \
                    );                                                          \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    auto inverse_perm = build_inverse_permutation(permutation);                 \
    return grad_permuted.view(permuted.sizes())                                 \
        .permute(inverse_perm)                                                  \
        .contiguous();                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                                         \
                                                                                \
    at::Tensor grad_grad_output = at::zeros_like(grad_output);                  \
    at::Tensor new_grad_input = at::zeros_like(arg);                            \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_grad_input_contig = grad_grad_input.contiguous();                 \
                                                                                \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg, dim);         \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES(                                             \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_all",                                 \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = grad_grad_input_contig.data_ptr<scalar_t>();\
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                                                                                \
                scalar_t gg_output;                                             \
                std::vector<scalar_t> new_grad(arg##_contig.numel());           \
                                                                                \
                kernel::NS::name##_backward_backward<scalar_t>(                 \
                    gg_input_ptr,                                               \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    __VA_OPT__(, __VA_ARGS__),                                  \
                    gg_output,                                                  \
                    new_grad.data()                                             \
                );                                                              \
                                                                                \
                grad_grad_output.fill_(gg_output);                              \
                                                                                \
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();   \
                for (int64_t i = 0; i < arg##_contig.numel(); ++i) {            \
                    new_grad_ptr[i] = new_grad[i];                              \
                }                                                               \
            }                                                                   \
        );                                                                      \
                                                                                \
        return std::make_tuple(grad_grad_output, new_grad_input);               \
    }                                                                           \
                                                                                \
    /* Dimension-specific backward_backward - return zeros for now */           \
    return std::make_tuple(grad_grad_output, new_grad_input);                   \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::NS */                                         \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {                                      \
    m.impl(#name, &torchscience::cpu::NS::name);                                \
    m.impl(#name "_backward", &torchscience::cpu::NS::name##_backward);         \
    m.impl(#name "_backward_backward", &torchscience::cpu::NS::name##_backward_backward);\
}

// =============================================================================
// FIXED REDUCTION MACROS
// =============================================================================

/**
 * CPU macro for unary fixed reduction operators.
 *
 * @param NS Namespace suffix
 * @param name Operator name
 * @param MODE ReductionMode::LAST_DIM or ReductionMode::ALL_DIMS
 * @param arg Tensor argument name
 * @param ... Extra arguments with types
 */
#define TORCHSCIENCE_CPU_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, MODE, arg, ...)\
namespace torchscience::cpu::NS {                                               \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                                         \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        auto output = at::empty({}, arg##_contig.options());                    \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_all",                                                   \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t result = kernel::NS::name<scalar_t>(                   \
                    data_ptr, arg##_contig.numel()                              \
                    __VA_OPT__(, __VA_ARGS__)                                   \
                );                                                              \
                output.fill_(result);                                           \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    } else {                                                                    \
        TORCH_CHECK(arg.dim() >= 1, #name ": input must have at least 1 dimension");\
                                                                                \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto output_shape = arg##_contig.sizes().vec();                         \
        output_shape.pop_back();                                                \
        auto output = output_shape.empty()                                      \
            ? at::empty({}, arg##_contig.options())                             \
            : at::empty(output_shape, arg##_contig.options());                  \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_cpu_last",                                                  \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* output_ptr = output.data_ptr<scalar_t>();             \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        output_ptr[b] = kernel::NS::name<scalar_t>(             \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            __VA_OPT__(, __VA_ARGS__)                           \
                        );                                                      \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    }                                                                           \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    at::Tensor grad_input = at::zeros_like(arg);                                \
    auto arg##_contig = arg.contiguous();                                       \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_all",                                          \
            [&]() {                                                             \
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                scalar_t* grad_ptr = grad_input.data_ptr<scalar_t>();           \
                                                                                \
                kernel::NS::name##_backward<scalar_t>(                          \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    __VA_OPT__(, __VA_ARGS__),                                  \
                    grad_ptr                                                    \
                );                                                              \
            }                                                                   \
        );                                                                      \
    } else {                                                                    \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto grad_output_flat = grad_output.contiguous().view({batch_size});    \
                                                                                \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_cpu_last",                                         \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();\
                scalar_t* grad_ptr = grad_input.data_ptr<scalar_t>();           \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        kernel::NS::name##_backward<scalar_t>(                  \
                            grad_out_ptr[b],                                    \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            __VA_OPT__(, __VA_ARGS__),                          \
                            grad_ptr + b * reduce_size                          \
                        );                                                      \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
    }                                                                           \
    return grad_input;                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    using namespace torchscience::cpu::reduction_detail;                                         \
                                                                                \
    at::Tensor grad_grad_output = at::zeros_like(grad_output);                  \
    at::Tensor new_grad_input = at::zeros_like(arg);                            \
                                                                                \
    auto arg##_contig = arg.contiguous();                                       \
    auto grad_grad_input_contig = grad_grad_input.contiguous();                 \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        AT_DISPATCH_FLOATING_TYPES(                                             \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_all",                                 \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = grad_grad_input_contig.data_ptr<scalar_t>();\
                scalar_t grad_out_val = grad_output.item<scalar_t>();           \
                                                                                \
                scalar_t gg_output;                                             \
                std::vector<scalar_t> new_grad(arg##_contig.numel());           \
                                                                                \
                kernel::NS::name##_backward_backward<scalar_t>(                 \
                    gg_input_ptr,                                               \
                    grad_out_val,                                               \
                    data_ptr,                                                   \
                    arg##_contig.numel()                                        \
                    __VA_OPT__(, __VA_ARGS__),                                  \
                    gg_output,                                                  \
                    new_grad.data()                                             \
                );                                                              \
                                                                                \
                grad_grad_output.fill_(gg_output);                              \
                                                                                \
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();   \
                for (int64_t i = 0; i < arg##_contig.numel(); ++i) {            \
                    new_grad_ptr[i] = new_grad[i];                              \
                }                                                               \
            }                                                                   \
        );                                                                      \
    } else {                                                                    \
        int64_t reduce_size = arg##_contig.size(-1);                            \
        int64_t batch_size = arg##_contig.numel() / reduce_size;                \
                                                                                \
        auto grad_output_flat = grad_output.contiguous().view({batch_size});    \
        auto gg_input_view = grad_grad_input_contig.view({batch_size, reduce_size});\
                                                                                \
        AT_DISPATCH_FLOATING_TYPES(                                             \
            arg##_contig.scalar_type(),                                         \
            #name "_backward_backward_cpu_last",                                \
            [&]() {                                                             \
                const scalar_t* data_ptr = arg##_contig.data_ptr<scalar_t>();   \
                const scalar_t* gg_input_ptr = gg_input_view.data_ptr<scalar_t>();\
                const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();\
                scalar_t* gg_output_ptr = grad_grad_output.data_ptr<scalar_t>();\
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();   \
                                                                                \
                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                    for (int64_t b = begin; b < end; ++b) {                     \
                        scalar_t gg_out;                                        \
                        kernel::NS::name##_backward_backward<scalar_t>(         \
                            gg_input_ptr + b * reduce_size,                     \
                            grad_out_ptr[b],                                    \
                            data_ptr + b * reduce_size,                         \
                            reduce_size                                         \
                            __VA_OPT__(, __VA_ARGS__),                          \
                            gg_out,                                             \
                            new_grad_ptr + b * reduce_size                      \
                        );                                                      \
                        gg_output_ptr[b] = gg_out;                              \
                    }                                                           \
                });                                                             \
            }                                                                   \
        );                                                                      \
    }                                                                           \
    return std::make_tuple(grad_grad_output, new_grad_input);                   \
}                                                                               \
                                                                                \
} /* namespace torchscience::cpu::NS */                                         \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {                                      \
    m.impl(#name, &torchscience::cpu::NS::name);                                \
    m.impl(#name "_backward", &torchscience::cpu::NS::name##_backward);         \
    m.impl(#name "_backward_backward", &torchscience::cpu::NS::name##_backward_backward);\
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/reduction_macros.h
git commit -m "feat(cpu): add dim-based and fixed reduction macros"
```

---

### Task 2: Create Meta Reduction Macros

**Files:**
- Create: `src/torchscience/csrc/meta/reduction_macros.h`

**Step 1: Create the Meta reduction macros header**

```cpp
// src/torchscience/csrc/meta/reduction_macros.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

// =============================================================================
// REDUCTION UTILITIES (internal to this header)
// =============================================================================

namespace torchscience::meta::reduction_detail {

inline std::vector<int64_t> compute_reduction_shape(
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
            TORCH_CHECK(pos_d >= 0 && pos_d < ndim, "Dimension out of range");
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

enum class ReductionMode {
    LAST_DIM,
    ALL_DIMS
};

}  // namespace torchscience::meta::reduction_detail

// =============================================================================
// DIM-BASED REDUCTION MACROS (Meta)
// =============================================================================

#define TORCHSCIENCE_META_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)      \
namespace torchscience::meta::NS {                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    using namespace torchscience::meta::reduction_detail;                       \
                                                                                \
    auto output_shape = compute_reduction_shape(arg, dim, keepdim);             \
                                                                                \
    auto output_dtype = at::isComplexType(arg.scalar_type())                    \
        ? c10::toRealValueType(arg.scalar_type())                               \
        : arg.scalar_type();                                                    \
                                                                                \
    if (output_shape.empty()) {                                                 \
        return at::empty({}, arg.options().dtype(output_dtype));                \
    }                                                                           \
    return at::empty(output_shape, arg.options().dtype(output_dtype));          \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    [[maybe_unused]] const at::Tensor& grad_output,                             \
    const at::Tensor& arg,                                                      \
    [[maybe_unused]] at::OptionalIntArrayRef dim,                               \
    [[maybe_unused]] bool keepdim                                               \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    return at::empty_like(arg);                                                 \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    [[maybe_unused]] const at::Tensor& grad_grad_input,                         \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    [[maybe_unused]] at::OptionalIntArrayRef dim,                               \
    [[maybe_unused]] bool keepdim                                               \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    return std::make_tuple(                                                     \
        at::empty_like(grad_output),                                            \
        at::empty_like(arg)                                                     \
    );                                                                          \
}                                                                               \
                                                                                \
} /* namespace torchscience::meta::NS */                                        \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, m) {                                     \
    m.impl(#name, &torchscience::meta::NS::name);                               \
    m.impl(#name "_backward", &torchscience::meta::NS::name##_backward);        \
    m.impl(#name "_backward_backward", &torchscience::meta::NS::name##_backward_backward);\
}

// =============================================================================
// FIXED REDUCTION MACROS (Meta)
// =============================================================================

#define TORCHSCIENCE_META_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, MODE, arg, ...)\
namespace torchscience::meta::NS {                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    using namespace torchscience::meta::reduction_detail;                       \
                                                                                \
    if constexpr (MODE == ReductionMode::ALL_DIMS) {                            \
        return at::empty({}, arg.options());                                    \
    } else {                                                                    \
        TORCH_CHECK(arg.dim() >= 1, #name ": input must have at least 1 dimension");\
        auto output_shape = arg.sizes().vec();                                  \
        output_shape.pop_back();                                                \
        if (output_shape.empty()) {                                             \
            return at::empty({}, arg.options());                                \
        }                                                                       \
        return at::empty(output_shape, arg.options());                          \
    }                                                                           \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    [[maybe_unused]] const at::Tensor& grad_output,                             \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    return at::empty_like(arg);                                                 \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    [[maybe_unused]] const at::Tensor& grad_grad_input,                         \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, [[maybe_unused]] __VA_ARGS__)                                  \
) {                                                                             \
    return std::make_tuple(                                                     \
        at::empty_like(grad_output),                                            \
        at::empty_like(arg)                                                     \
    );                                                                          \
}                                                                               \
                                                                                \
} /* namespace torchscience::meta::NS */                                        \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Meta, m) {                                     \
    m.impl(#name, &torchscience::meta::NS::name);                               \
    m.impl(#name "_backward", &torchscience::meta::NS::name##_backward);        \
    m.impl(#name "_backward_backward", &torchscience::meta::NS::name##_backward_backward);\
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/reduction_macros.h
git commit -m "feat(meta): add dim-based and fixed reduction macros"
```

---

### Task 3: Create Autograd Reduction Macros

**Files:**
- Create: `src/torchscience/csrc/autograd/reduction_macros.h`

**Step 1: Create the Autograd reduction macros header**

```cpp
// src/torchscience/csrc/autograd/reduction_macros.h
#pragma once

#include <vector>

#include <torch/extension.h>

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autograd)
// =============================================================================

#define TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg, ...)\
namespace torchscience::autograd::NS {                                          \
                                                                                \
class Name##Backward : public torch::autograd::Function<Name##Backward> {       \
public:                                                                         \
    static std::vector<at::Tensor> forward(                                     \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& grad_output,                                          \
        const at::Tensor& arg,                                                  \
        std::vector<int64_t> dim_vec,                                           \
        bool keepdim,                                                           \
        bool arg##_requires_grad                                                \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
        /* Save extra args in saved_data here if needed */                      \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name "_backward", "")          \
            .typed<at::Tensor(                                                  \
                const at::Tensor&, const at::Tensor&,                           \
                at::OptionalIntArrayRef, bool                                   \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg, dim_ref, keepdim                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)))};     \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};\
        }                                                                       \
                                                                                \
        auto dim_vec = ctx->saved_data["dim"].toIntVector();                    \
        bool keepdim = ctx->saved_data["keepdim"].toBool();                     \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        auto [gg_output, new_grad] = c10::Dispatcher::singleton()               \
            .findSchemaOrThrow("torchscience::" #name "_backward_backward", "") \
            .typed<std::tuple<at::Tensor, at::Tensor>(                          \
                const at::Tensor&, const at::Tensor&, const at::Tensor&,        \
                at::OptionalIntArrayRef, bool                                   \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1], dim_ref, keepdim         \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));  \
                                                                                \
        return {gg_output, new_grad, at::Tensor(), at::Tensor(), at::Tensor()}; \
    }                                                                           \
};                                                                              \
                                                                                \
class Name : public torch::autograd::Function<Name> {                           \
public:                                                                         \
    static at::Tensor forward(                                                  \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& arg,                                                  \
        std::vector<int64_t> dim_vec,                                           \
        bool keepdim                                                            \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name, "")                      \
            .typed<at::Tensor(                                                  \
                const at::Tensor&, at::OptionalIntArrayRef, bool                \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>();                                                               \
                                                                                \
        return op.call(arg, dim_ref, keepdim                                    \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));      \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
        torch::autograd::AutogradContext* ctx,                                  \
        const torch::autograd::variable_list& grad_outputs                      \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        auto dim_vec = ctx->saved_data["dim"].toIntVector();                    \
        bool keepdim = ctx->saved_data["keepdim"].toBool();                     \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], dim_vec, keepdim, arg##_requires_grad    \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__))        \
        );                                                                      \
                                                                                \
        return {                                                                \
            arg##_requires_grad ? grads[0] : at::Tensor(),                      \
            at::Tensor(),  /* dim */                                            \
            at::Tensor()   /* keepdim */                                        \
            /* extra args return at::Tensor() */                                \
        };                                                                      \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    std::vector<int64_t> dim_vec;                                               \
    if (dim.has_value()) {                                                      \
        dim_vec = dim->vec();                                                   \
    }                                                                           \
    return Name::apply(arg, dim_vec, keepdim                                    \
        __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));          \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}

// Helper macros for extracting types and names from variadic args
// These need refinement based on actual usage patterns
#define TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(...) __VA_ARGS__
#define TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(...) __VA_ARGS__

// =============================================================================
// FIXED REDUCTION MACROS (Autograd)
// =============================================================================

#define TORCHSCIENCE_AUTOGRAD_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg, ...)\
namespace torchscience::autograd::NS {                                          \
                                                                                \
class Name##Backward : public torch::autograd::Function<Name##Backward> {       \
public:                                                                         \
    static std::vector<at::Tensor> forward(                                     \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& grad_output,                                          \
        const at::Tensor& arg,                                                  \
        bool arg##_requires_grad                                                \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name "_backward", "")          \
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&              \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg                                        \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)))};     \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor()};                  \
        }                                                                       \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        auto [gg_output, new_grad] = c10::Dispatcher::singleton()               \
            .findSchemaOrThrow("torchscience::" #name "_backward_backward", "") \
            .typed<std::tuple<at::Tensor, at::Tensor>(                          \
                const at::Tensor&, const at::Tensor&, const at::Tensor&         \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1]                           \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));  \
                                                                                \
        return {gg_output, new_grad, at::Tensor()};                             \
    }                                                                           \
};                                                                              \
                                                                                \
class Name : public torch::autograd::Function<Name> {                           \
public:                                                                         \
    static at::Tensor forward(                                                  \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& arg                                                   \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name, "")                      \
            .typed<at::Tensor(const at::Tensor&                                 \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_TYPES(__VA_ARGS__))    \
            )>();                                                               \
                                                                                \
        return op.call(arg                                                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));      \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
        torch::autograd::AutogradContext* ctx,                                  \
        const torch::autograd::variable_list& grad_outputs                      \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], arg##_requires_grad                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__))        \
        );                                                                      \
                                                                                \
        return {arg##_requires_grad ? grads[0] : at::Tensor()};                 \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    return Name::apply(arg                                                      \
        __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_EXTRA_NAMES(__VA_ARGS__)));          \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autograd/reduction_macros.h
git commit -m "feat(autograd): add dim-based and fixed reduction macros"
```

---

### Task 4: Create Autocast Reduction Macros

**Files:**
- Create: `src/torchscience/csrc/autocast/reduction_macros.h`

**Step 1: Create the Autocast reduction macros header**

```cpp
// src/torchscience/csrc/autocast/reduction_macros.h
#pragma once

#include <tuple>

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autocast)
// =============================================================================

#define TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)  \
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, at::OptionalIntArrayRef, bool                    \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, arg),                           \
        dim,                                                                    \
        keepdim                                                                 \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, const at::Tensor&,                               \
            at::OptionalIntArrayRef, bool                                       \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, grad_output),                   \
        at::autocast::cached_cast(target_dtype, arg),                           \
        dim,                                                                    \
        keepdim                                                                 \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&,            \
            at::OptionalIntArrayRef, bool                                       \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, grad_grad_input),               \
        at::autocast::cached_cast(target_dtype, grad_output),                   \
        at::autocast::cached_cast(target_dtype, arg),                           \
        dim,                                                                    \
        keepdim                                                                 \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}

// Helper macros
#define TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(...) __VA_ARGS__
#define TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(...) __VA_ARGS__

// =============================================================================
// FIXED REDUCTION MACROS (Autocast)
// =============================================================================

#define TORCHSCIENCE_AUTOCAST_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)  \
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(const at::Tensor&                                     \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, arg)                            \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&                  \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, grad_output),                   \
        at::autocast::cached_cast(target_dtype, arg)                            \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    static auto op = c10::Dispatcher::singleton()                               \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&             \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_TYPES(__VA_ARGS__))        \
        )>();                                                                   \
                                                                                \
    auto target_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
                                                                                \
    return op.call(                                                             \
        at::autocast::cached_cast(target_dtype, grad_grad_input),               \
        at::autocast::cached_cast(target_dtype, grad_output),                   \
        at::autocast::cached_cast(target_dtype, arg)                            \
        __VA_OPT__(, TORCHSCIENCE_AUTOCAST_EXTRA_NAMES(__VA_ARGS__))            \
    );                                                                          \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/autocast/reduction_macros.h
git commit -m "feat(autocast): add dim-based and fixed reduction macros"
```

---

## Phase 2: Validation with Test Operator

### Task 5: Create Test Kernel for Validation

Create a simple "sum_squares" reduction operator to validate the macros work correctly before migrating kurtosis.

**Files:**
- Create: `src/torchscience/csrc/kernel/test/sum_squares.h`

**Step 1: Create the test kernel**

```cpp
// src/torchscience/csrc/kernel/test/sum_squares.h
#pragma once

#include <cmath>

namespace torchscience::kernel::test {

/**
 * Simple sum of squares reduction for testing macros.
 * f(x) = sum(x_i^2)
 */
template <typename T>
T sum_squares(const T* data, int64_t n) {
    T result = T(0);
    for (int64_t i = 0; i < n; ++i) {
        result += data[i] * data[i];
    }
    return result;
}

/**
 * Backward: df/dx_i = 2 * x_i
 */
template <typename T>
void sum_squares_backward(
    T grad_output,
    const T* data,
    int64_t n,
    T* grad_input
) {
    for (int64_t i = 0; i < n; ++i) {
        grad_input[i] = grad_output * T(2) * data[i];
    }
}

/**
 * Backward-backward: d2f/dx_i dx_j = 2 * delta_ij
 */
template <typename T>
void sum_squares_backward_backward(
    const T* grad_grad_input,
    T grad_output,
    const T* data,
    int64_t n,
    T& grad_grad_output,
    T* new_grad_input
) {
    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
        // d(grad_input_i)/d(grad_output) = 2 * x_i
        grad_grad_output += grad_grad_input[i] * T(2) * data[i];
        // d(grad_input_i)/d(x_j) = 2 * grad_output * delta_ij
        new_grad_input[i] = grad_output * T(2) * grad_grad_input[i];
    }
}

}  // namespace torchscience::kernel::test
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/test/sum_squares.h
git commit -m "test: add sum_squares kernel for reduction macro validation"
```

---

### Task 6: Create Test Operator Using Macros

**Files:**
- Create: `src/torchscience/csrc/cpu/test/sum_squares.h`
- Create: `src/torchscience/csrc/meta/test/sum_squares.h`

**Step 1: Create CPU test operator**

```cpp
// src/torchscience/csrc/cpu/test/sum_squares.h
#pragma once

#include "torchscience/csrc/cpu/reduction_macros.h"
#include "torchscience/csrc/kernel/test/sum_squares.h"

TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR(
    test,
    sum_squares,
    input
)
```

**Step 2: Create Meta test operator**

```cpp
// src/torchscience/csrc/meta/test/sum_squares.h
#pragma once

#include "torchscience/csrc/meta/reduction_macros.h"

TORCHSCIENCE_META_DIM_REDUCTION_UNARY_OPERATOR(
    test,
    sum_squares,
    input
)
```

**Step 3: Add schema to torchscience.cpp**

Add after existing schema definitions:

```cpp
// Test reduction operator
module.def("sum_squares(Tensor input, int[]? dim, bool keepdim) -> Tensor");
module.def("sum_squares_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> Tensor");
module.def("sum_squares_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> (Tensor, Tensor)");
```

**Step 4: Add includes to torchscience.cpp**

```cpp
#include "cpu/test/sum_squares.h"
#include "meta/test/sum_squares.h"
```

**Step 5: Commit**

```bash
git add src/torchscience/csrc/cpu/test/sum_squares.h
git add src/torchscience/csrc/meta/test/sum_squares.h
git add src/torchscience/csrc/torchscience.cpp
git commit -m "test: add sum_squares operator using reduction macros"
```

---

### Task 7: Write Tests for Reduction Macros

**Files:**
- Create: `tests/torchscience/test__reduction_macros.py`

**Step 1: Create the test file**

```python
"""Tests for reduction operator macro infrastructure."""

import pytest
import torch

# Import will fail until we add Python bindings
# For now, test via ops directly
import torchscience._csrc


class TestSumSquaresForward:
    """Test sum_squares forward pass."""

    def test_1d_all_dims(self):
        """Test reducing all dims of 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = (1**2 + 2**2 + 3**2)
        assert result.shape == ()
        assert torch.allclose(result, torch.tensor(expected))

    def test_2d_all_dims(self):
        """Test reducing all dims of 2D tensor."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        expected = 1 + 4 + 9 + 16
        assert result.shape == ()
        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float))

    def test_2d_dim0(self):
        """Test reducing along dim 0."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [0], False)
        expected = torch.tensor([1 + 9, 4 + 16], dtype=torch.float)
        assert result.shape == (2,)
        assert torch.allclose(result, expected)

    def test_2d_dim1(self):
        """Test reducing along dim 1."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], False)
        expected = torch.tensor([1 + 4, 9 + 16], dtype=torch.float)
        assert result.shape == (2,)
        assert torch.allclose(result, expected)

    def test_keepdim_true(self):
        """Test keepdim=True."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], True)
        assert result.shape == (2, 1)


class TestSumSquaresBackward:
    """Test sum_squares backward pass."""

    def test_gradient_all_dims(self):
        """Test gradient when reducing all dims."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        op = torch.ops.torchscience.sum_squares
        result = op(x, None, False)
        result.backward()
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad)

    def test_gradient_single_dim(self):
        """Test gradient when reducing single dim."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        op = torch.ops.torchscience.sum_squares
        result = op(x, [1], False)
        result.sum().backward()
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad)


class TestSumSquaresGradcheck:
    """Test gradients with torch.autograd.gradcheck."""

    def test_gradcheck_all_dims(self):
        """Gradcheck for all dims reduction."""
        x = torch.randn(5, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, None, False)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradcheck_single_dim(self):
        """Gradcheck for single dim reduction."""
        x = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, [1], False)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_gradgradcheck_all_dims(self):
        """Second-order gradient check."""
        x = torch.randn(5, requires_grad=True, dtype=torch.float64)
        op = torch.ops.torchscience.sum_squares

        def fn(t):
            return op(t, None, False)

        assert torch.autograd.gradgradcheck(fn, (x,), raise_exception=True)
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/torchscience/test__reduction_macros.py -v`

Expected: Tests should pass once the macros compile correctly

**Step 3: Commit**

```bash
git add tests/torchscience/test__reduction_macros.py
git commit -m "test: add tests for reduction macro infrastructure"
```

---

## Phase 3: Binary and Higher-Arity Macros

### Task 8: Add Binary Dim-Reduction Macros

**Files:**
- Modify: `src/torchscience/csrc/cpu/reduction_macros.h`
- Modify: `src/torchscience/csrc/meta/reduction_macros.h`

**Step 1: Add binary macro to CPU file**

Add after the unary macro:

```cpp
/**
 * CPU macro for binary dim-based reduction operators.
 * Both inputs must have the same reduction dimension size.
 */
#define TORCHSCIENCE_CPU_DIM_REDUCTION_BINARY_OPERATOR(NS, name, arg1, arg2, ...)\
namespace torchscience::cpu::NS {                                               \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg1,                                                     \
    const at::Tensor& arg2,                                                     \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    TORCH_CHECK(arg1.numel() > 0 && arg2.numel() > 0,                           \
        #name ": input tensors must be non-empty");                             \
                                                                                \
    using namespace torchscience::cpu::reduction_detail;                        \
                                                                                \
    /* Broadcast inputs to common shape */                                      \
    auto [arg1##_expanded, arg2##_expanded] = at::broadcast_tensors({arg1, arg2});\
                                                                                \
    auto output_shape = compute_reduction_shape(arg1##_expanded, dim, keepdim); \
    auto [reduce_size, batch_size] = compute_reduction_sizes(arg1##_expanded, dim);\
                                                                                \
    auto arg1##_contig = arg1##_expanded.contiguous();                          \
    auto arg2##_contig = arg2##_expanded.contiguous();                          \
                                                                                \
    auto output_dtype = at::result_type(arg1, arg2);                            \
    auto options = arg1##_contig.options().dtype(output_dtype);                 \
    at::Tensor output = output_shape.empty()                                    \
        ? at::empty({}, options)                                                \
        : at::empty(output_shape, options);                                     \
                                                                                \
    if (!dim.has_value() || dim->empty()) {                                     \
        AT_DISPATCH_FLOATING_TYPES_AND2(                                        \
            at::kBFloat16, at::kHalf,                                           \
            output_dtype,                                                       \
            #name "_cpu_all",                                                   \
            [&]() {                                                             \
                const scalar_t* data1_ptr = arg1##_contig.data_ptr<scalar_t>(); \
                const scalar_t* data2_ptr = arg2##_contig.data_ptr<scalar_t>(); \
                scalar_t result = kernel::NS::name<scalar_t>(                   \
                    data1_ptr, data2_ptr, arg1##_contig.numel()                 \
                    __VA_OPT__(, __VA_ARGS__)                                   \
                );                                                              \
                output.fill_(result);                                           \
            }                                                                   \
        );                                                                      \
        return output;                                                          \
    }                                                                           \
                                                                                \
    auto permutation = build_reduction_permutation(arg1##_expanded.dim(), dim); \
    auto permuted1 = arg1##_contig.permute(permutation).contiguous();           \
    auto permuted2 = arg2##_contig.permute(permutation).contiguous();           \
    auto permuted_view1 = permuted1.view({batch_size, reduce_size});            \
    auto permuted_view2 = permuted2.view({batch_size, reduce_size});            \
                                                                                \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
        at::kBFloat16, at::kHalf,                                               \
        output_dtype,                                                           \
        #name "_cpu_dim",                                                       \
        [&]() {                                                                 \
            const scalar_t* data1_ptr = permuted_view1.data_ptr<scalar_t>();    \
            const scalar_t* data2_ptr = permuted_view2.data_ptr<scalar_t>();    \
            scalar_t* output_ptr = output.data_ptr<scalar_t>();                 \
                                                                                \
            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {\
                for (int64_t b = begin; b < end; ++b) {                         \
                    output_ptr[b] = kernel::NS::name<scalar_t>(                 \
                        data1_ptr + b * reduce_size,                            \
                        data2_ptr + b * reduce_size,                            \
                        reduce_size                                             \
                        __VA_OPT__(, __VA_ARGS__)                               \
                    );                                                          \
                }                                                               \
            });                                                                 \
        }                                                                       \
    );                                                                          \
                                                                                \
    return output;                                                              \
}                                                                               \
                                                                                \
/* backward and backward_backward follow similar pattern... */                  \
                                                                                \
} /* namespace torchscience::cpu::NS */                                         \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {                                      \
    m.impl(#name, &torchscience::cpu::NS::name);                                \
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/reduction_macros.h
git add src/torchscience/csrc/meta/reduction_macros.h
git commit -m "feat: add binary dim-reduction macros"
```

---

## Phase 4: Migration

### Task 9: Migrate Kurtosis to Use Macros

This task migrates the existing kurtosis implementation to use the new macro infrastructure.

**Files:**
- Create: `src/torchscience/csrc/kernel/statistics/descriptive/kurtosis.h`
- Modify: `src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h`
- Modify: `src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h`
- Modify: `src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h`
- Modify: `src/torchscience/csrc/autocast/statistics/descriptive/kurtosis.h`

**Step 1: Extract kernel from existing kurtosis implementation**

Create `src/torchscience/csrc/kernel/statistics/descriptive/kurtosis.h` by extracting the `kurtosis_1d`, `kurtosis_backward_1d`, and `kurtosis_backward_backward_1d` template functions from the existing CPU implementation.

**Step 2: Refactor CPU kurtosis to use macro**

Replace the ~970 lines with:

```cpp
// src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h
#pragma once

#include "torchscience/csrc/cpu/reduction_macros.h"
#include "torchscience/csrc/kernel/statistics/descriptive/kurtosis.h"

TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR(
    statistics::descriptive,
    kurtosis,
    input,
    bool fisher, bool bias
)
```

**Step 3: Run tests to verify behavior unchanged**

Run: `uv run pytest tests/torchscience/stats/descriptive/test__kurtosis.py -v`

Expected: All existing tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/kernel/statistics/descriptive/kurtosis.h
git add src/torchscience/csrc/cpu/statistics/descriptive/kurtosis.h
git add src/torchscience/csrc/meta/statistics/descriptive/kurtosis.h
git add src/torchscience/csrc/autograd/statistics/descriptive/kurtosis.h
git add src/torchscience/csrc/autocast/statistics/descriptive/kurtosis.h
git commit -m "refactor(kurtosis): migrate to reduction macro infrastructure"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-4 | Core infrastructure: CPU, Meta, Autograd, Autocast macros (utilities inline) |
| 2 | 5-7 | Validation with test operator (sum_squares) |
| 3 | 8 | Binary and higher-arity macros |
| 4 | 9 | Migrate kurtosis to use macros |

**Total new files:** ~7
**Lines of macro infrastructure:** ~1500
**Lines saved per operator after migration:** ~800-900

**Future work:**
- Add ternary and quaternary variants
- Add CUDA reduction macros
- Migrate rosenbrock, cook_torrance, histogram
- Add complex type support to reduction macros
