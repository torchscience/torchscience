#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::descriptive {

namespace {

/**
 * Compute output shape after reduction.
 */
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
    [[maybe_unused]] const at::Tensor& gradientient_output,
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
