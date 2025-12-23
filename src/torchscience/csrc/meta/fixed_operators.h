#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta {

// =============================================================================
// MetaFixedOperator - Shape inference for fixed operators
// =============================================================================

template<typename FixedTraits>
struct MetaFixedOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        int64_t dim,
        Args... args
    ) {
        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;
        TORCH_CHECK(dim >= 0 && dim < ndim, "fixed_op: dim out of range");

        int64_t input_size = input.size(dim);
        int64_t output_size = FixedTraits::output_size(input_size, args...);

        std::vector<int64_t> output_shape = input.sizes().vec();
        output_shape[dim] = output_size;

        return at::empty(output_shape, input.options());
    }

    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t dim,
        Args...
    ) {
        (void)grad_output;
        (void)dim;
        return at::empty_like(input);
    }

    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t dim,
        Args...
    ) {
        (void)grad_grad_input;
        (void)dim;
        return std::make_tuple(
            at::empty_like(grad_output),
            at::empty_like(input)
        );
    }
};

}  // namespace torchscience::meta
