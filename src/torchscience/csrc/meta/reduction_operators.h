#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta {

namespace reduction_detail {

inline std::vector<int64_t> compute_output_shape(
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

}  // namespace reduction_detail

// =============================================================================
// MetaReductionOperator - Shape inference for reduction operators
// =============================================================================

struct MetaReductionOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args...
    ) {
        auto output_shape = reduction_detail::compute_output_shape(input, dim, keepdim);

        if (output_shape.empty()) {
            return at::empty({}, input.options());
        }
        return at::empty(output_shape, input.options());
    }

    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args...
    ) {
        (void)grad_output;
        (void)dim;
        (void)keepdim;
        return at::empty_like(input);
    }

    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args...
    ) {
        (void)grad_grad_input;
        (void)dim;
        (void)keepdim;
        return std::make_tuple(
            at::empty_like(grad_output),
            at::empty_like(input)
        );
    }
};

}  // namespace torchscience::meta
