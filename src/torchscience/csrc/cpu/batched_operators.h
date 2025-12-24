#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUBatchedOperator - Template for operators with fixed trailing dimensions
// =============================================================================

// BatchedTraits must provide:
//   - static constexpr int64_t num_trailing_dims;  // e.g., 2 for matrix ops
//   - static std::vector<int64_t> output_trailing_shape(ArrayRef<int64_t> input_trailing);
//   - template<T> static void kernel(const T* in, T* out, ArrayRef<int64_t> inner_shape);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, ArrayRef<int64_t> inner_shape);
//   - template<T> static void backward_backward_kernel(
//         const T* grad_grad_in, const T* grad_out, const T* in,
//         ArrayRef<int64_t> inner_shape, T* grad_grad_out, T* new_grad_in);

template<typename BatchedTraits>
struct CPUBatchedOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        Args... args
    ) {
        TORCH_CHECK(input.numel() > 0, "batched_op: input tensor must be non-empty");

        constexpr int64_t num_trailing = BatchedTraits::num_trailing_dims;
        int64_t ndim = input.dim();
        TORCH_CHECK(ndim >= num_trailing,
            "batched_op: input must have at least ", num_trailing, " dimensions");

        // Extract trailing dimensions
        std::vector<int64_t> trailing_shape;
        for (int64_t i = ndim - num_trailing; i < ndim; ++i) {
            trailing_shape.push_back(input.size(i));
        }

        // Compute output trailing shape
        std::vector<int64_t> output_trailing = BatchedTraits::output_trailing_shape(trailing_shape);

        // Compute batch size
        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - num_trailing; ++i) {
            batch_size *= input.size(i);
        }

        // Compute inner sizes
        int64_t inner_input_size = 1;
        for (int64_t s : trailing_shape) {
            inner_input_size *= s;
        }
        int64_t inner_output_size = 1;
        for (int64_t s : output_trailing) {
            inner_output_size *= s;
        }

        // Build output shape
        std::vector<int64_t> output_shape;
        for (int64_t i = 0; i < ndim - num_trailing; ++i) {
            output_shape.push_back(input.size(i));
        }
        for (int64_t s : output_trailing) {
            output_shape.push_back(s);
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor output = at::empty(output_shape, input.options());

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "batched_cpu",
            [&]() {
                const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* out_ptr = output.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        BatchedTraits::template kernel<scalar_t>(
                            in_ptr + b * inner_input_size,
                            out_ptr + b * inner_output_size,
                            trailing_shape,
                            args...
                        );
                    }
                });
            }
        );

        return output;
    }

    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        Args... args
    ) {
        constexpr int64_t num_trailing = BatchedTraits::num_trailing_dims;
        int64_t ndim = input.dim();

        std::vector<int64_t> trailing_shape;
        for (int64_t i = ndim - num_trailing; i < ndim; ++i) {
            trailing_shape.push_back(input.size(i));
        }

        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - num_trailing; ++i) {
            batch_size *= input.size(i);
        }

        int64_t inner_input_size = 1;
        for (int64_t s : trailing_shape) {
            inner_input_size *= s;
        }

        std::vector<int64_t> output_trailing = BatchedTraits::output_trailing_shape(trailing_shape);
        int64_t inner_output_size = 1;
        for (int64_t s : output_trailing) {
            inner_output_size *= s;
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_input = at::zeros_like(input);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "batched_backward_cpu",
            [&]() {
                const scalar_t* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
                const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        BatchedTraits::template backward_kernel<scalar_t>(
                            grad_out_ptr + b * inner_output_size,
                            in_ptr + b * inner_input_size,
                            grad_in_ptr + b * inner_input_size,
                            trailing_shape,
                            args...
                        );
                    }
                });
            }
        );

        return grad_input;
    }

    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        Args... args
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        constexpr int64_t num_trailing = BatchedTraits::num_trailing_dims;
        int64_t ndim = input.dim();

        std::vector<int64_t> trailing_shape;
        for (int64_t i = ndim - num_trailing; i < ndim; ++i) {
            trailing_shape.push_back(input.size(i));
        }

        std::vector<int64_t> output_trailing = BatchedTraits::output_trailing_shape(trailing_shape);

        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - num_trailing; ++i) {
            batch_size *= input.size(i);
        }

        int64_t inner_input_size = 1;
        for (int64_t s : trailing_shape) {
            inner_input_size *= s;
        }
        int64_t inner_output_size = 1;
        for (int64_t s : output_trailing) {
            inner_output_size *= s;
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        at::Tensor grad_grad_output = at::zeros_like(grad_output);
        at::Tensor new_grad_input = at::zeros_like(input);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "batched_backward_backward_cpu",
            [&]() {
                const scalar_t* gg_in_ptr = grad_grad_input_contig.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
                const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* gg_out_ptr = grad_grad_output.data_ptr<scalar_t>();
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        BatchedTraits::template backward_backward_kernel<scalar_t>(
                            gg_in_ptr + b * inner_input_size,
                            grad_out_ptr + b * inner_output_size,
                            in_ptr + b * inner_input_size,
                            trailing_shape,
                            gg_out_ptr + b * inner_output_size,
                            new_grad_ptr + b * inner_input_size,
                            args...
                        );
                    }
                });
            }
        );

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
};

}  // namespace torchscience::cpu
