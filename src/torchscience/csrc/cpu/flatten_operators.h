#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUFlattenOperator - Template for operators that flatten input
// =============================================================================

// FlattenTraits must provide:
//   - static std::vector<int64_t> output_shape(int64_t numel, Args... args);
//   - template<T> static void kernel(const T* in, int64_t numel, T* out, int64_t out_size, Args...);

template<typename FlattenTraits>
struct CPUFlattenOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        Args... args
    ) {
        TORCH_CHECK(input.numel() > 0, "flatten_op: input tensor must be non-empty");

        int64_t numel = input.numel();
        std::vector<int64_t> output_shape = FlattenTraits::output_shape(numel, args...);

        int64_t output_size = 1;
        for (int64_t s : output_shape) {
            output_size *= s;
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor output = at::empty(output_shape, input.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "flatten_cpu",
            [&]() {
                const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();

                FlattenTraits::template kernel<scalar_t>(
                    in_ptr, numel, out_ptr, output_size, args...
                );
            }
        );

        return output;
    }

    // Backward for flatten ops is often complex or non-differentiable
    // Default implementation returns zeros
    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        Args... args
    ) {
        (void)grad_output;
        (void)args;
        return at::zeros_like(input);
    }
};

}  // namespace torchscience::cpu
