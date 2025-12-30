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
