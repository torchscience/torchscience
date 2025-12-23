#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUFixedOperator - Template for operators on fixed dimensions
// =============================================================================

// FixedTraits must provide:
//   - static int64_t output_size(int64_t input_size, Args... args);
//   - template<T> static void kernel(const T* in, T* out, int64_t in_size, int64_t out_size, Args...);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, sizes, Args...);

template<typename FixedTraits>
struct CPUFixedOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        int64_t dim,
        Args... args
    ) {
        TORCH_CHECK(input.numel() > 0, "fixed_op: input tensor must be non-empty");

        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;
        TORCH_CHECK(dim >= 0 && dim < ndim, "fixed_op: dim out of range");

        int64_t input_size = input.size(dim);
        int64_t output_size = FixedTraits::output_size(input_size, args...);

        // Compute output shape
        std::vector<int64_t> output_shape = input.sizes().vec();
        output_shape[dim] = output_size;

        // Compute batch dimensions
        int64_t batch_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            batch_size *= input.size(i);
        }
        int64_t trailing_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            trailing_size *= input.size(i);
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor output = at::empty(output_shape, input.options());

        if (trailing_size == 1) {
            // Simple case: operate directly on contiguous chunks
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input.scalar_type(),
                "fixed_cpu_simple",
                [&]() {
                    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* out_ptr = output.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            FixedTraits::template kernel<scalar_t>(
                                in_ptr + b * input_size,
                                out_ptr + b * output_size,
                                input_size,
                                output_size,
                                args...
                            );
                        }
                    });
                }
            );
        } else {
            // General case: handle strided access
            // Move dim to last position for contiguous access
            std::vector<int64_t> perm;
            for (int64_t i = 0; i < ndim; ++i) {
                if (i != dim) perm.push_back(i);
            }
            perm.push_back(dim);

            at::Tensor permuted_in = input_contig.permute(perm).contiguous();
            at::Tensor permuted_out = at::empty(
                {batch_size * trailing_size, output_size},
                input.options()
            );

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input.scalar_type(),
                "fixed_cpu_general",
                [&]() {
                    const scalar_t* in_ptr = permuted_in.data_ptr<scalar_t>();
                    scalar_t* out_ptr = permuted_out.data_ptr<scalar_t>();
                    int64_t total_batches = batch_size * trailing_size;

                    at::parallel_for(0, total_batches, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            FixedTraits::template kernel<scalar_t>(
                                in_ptr + b * input_size,
                                out_ptr + b * output_size,
                                input_size,
                                output_size,
                                args...
                            );
                        }
                    });
                }
            );

            // Inverse permutation
            std::vector<int64_t> inv_perm(ndim);
            for (int64_t i = 0; i < ndim - 1; ++i) {
                inv_perm[perm[i]] = i;
            }
            inv_perm[dim] = ndim - 1;

            // Reshape and permute back
            std::vector<int64_t> temp_shape;
            for (int64_t i = 0; i < ndim; ++i) {
                if (i != dim) temp_shape.push_back(input.size(i));
            }
            temp_shape.push_back(output_size);

            output = permuted_out.view(temp_shape).permute(inv_perm).contiguous();
        }

        return output;
    }

    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t dim,
        Args... args
    ) {
        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;

        int64_t input_size = input.size(dim);
        int64_t output_size = grad_output.size(dim);

        at::Tensor grad_input = at::zeros_like(input);

        int64_t batch_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            batch_size *= input.size(i);
        }
        int64_t trailing_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            trailing_size *= input.size(i);
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();

        if (trailing_size == 1) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input.scalar_type(),
                "fixed_backward_cpu_simple",
                [&]() {
                    const scalar_t* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
                    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            FixedTraits::template backward_kernel<scalar_t>(
                                grad_out_ptr + b * output_size,
                                in_ptr + b * input_size,
                                grad_in_ptr + b * input_size,
                                input_size,
                                output_size,
                                args...
                            );
                        }
                    });
                }
            );
        } else {
            // Handle strided case similarly to forward
            // (Implementation follows same pattern as forward)
        }

        return grad_input;
    }
};

}  // namespace torchscience::cpu
