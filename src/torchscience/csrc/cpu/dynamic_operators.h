#pragma once

#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUDynamicOperator - Template for operators with data-dependent output shape
// =============================================================================

// DynamicTraits must provide:
//   - template<T> static std::vector<at::Tensor> kernel(const at::Tensor& input, Args... args);
//
// Note: Dynamic operators typically cannot use TensorIterator since output shape
// is unknown until computation completes. They also often have limited or no
// autograd support.

template<typename DynamicTraits>
struct CPUDynamicOperator {
    template<typename... Args>
    static std::vector<at::Tensor> forward(
        const at::Tensor& input,
        Args... args
    ) {
        TORCH_CHECK(input.numel() > 0, "dynamic_op: input tensor must be non-empty");

        at::Tensor input_contig = input.contiguous();

        std::vector<at::Tensor> result;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "dynamic_cpu",
            [&]() {
                result = DynamicTraits::template kernel<scalar_t>(input_contig, args...);
            }
        );

        return result;
    }

    // Dynamic operators are typically non-differentiable
    // If differentiation is needed, use a relaxation or surrogate gradient
    template<typename... Args>
    static at::Tensor backward(
        const std::vector<at::Tensor>& grad_outputs,
        const at::Tensor& input,
        Args... args
    ) {
        (void)grad_outputs;
        (void)args;
        // Return zeros - most dynamic ops are non-differentiable
        return at::zeros_like(input);
    }
};

}  // namespace torchscience::cpu
