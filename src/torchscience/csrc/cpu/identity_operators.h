#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

// =============================================================================
// CPUIdentityOperator - Template for shape-preserving operators
// =============================================================================

// IdentityTraits must provide:
//   - static constexpr int64_t channel_size;  // e.g., 3 for RGB
//   - template<T> static void kernel(const T* in, T* out, int64_t channel_size);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, int64_t channel_size);
//   - template<T> static void backward_backward_kernel(
//         const T* grad_grad_in, const T* grad_out, const T* in, int64_t channel_size,
//         T* grad_grad_out, T* new_grad_in);

template<typename IdentityTraits>
struct CPUIdentityOperator {
    static at::Tensor forward(
        const at::Tensor& input,
        int64_t channel_dim = -1
    ) {
        TORCH_CHECK(input.numel() > 0, "identity_op: input tensor must be non-empty");

        int64_t ndim = input.dim();
        if (channel_dim < 0) channel_dim += ndim;
        TORCH_CHECK(channel_dim >= 0 && channel_dim < ndim, "identity_op: channel_dim out of range");

        int64_t channel_size = input.size(channel_dim);
        TORCH_CHECK(channel_size == IdentityTraits::channel_size,
            "identity_op: expected channel size ", IdentityTraits::channel_size,
            " but got ", channel_size);

        // Compute batch size
        int64_t batch_size = input.numel() / channel_size;

        at::Tensor input_contig = input.contiguous();
        at::Tensor output = at::empty_like(input);

        // Move channel dim to last for efficient access
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != channel_dim) perm.push_back(i);
        }
        perm.push_back(channel_dim);

        at::Tensor permuted_in = input_contig.permute(perm).contiguous();
        at::Tensor permuted_out = at::empty_like(permuted_in);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "identity_cpu",
            [&]() {
                const scalar_t* in_ptr = permuted_in.data_ptr<scalar_t>();
                scalar_t* out_ptr = permuted_out.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        IdentityTraits::template kernel<scalar_t>(
                            in_ptr + b * channel_size,
                            out_ptr + b * channel_size,
                            channel_size
                        );
                    }
                });
            }
        );

        // Inverse permutation
        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
            inv_perm[perm[i]] = i;
        }

        output = permuted_out.permute(inv_perm).contiguous();
        return output;
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t channel_dim = -1
    ) {
        int64_t ndim = input.dim();
        if (channel_dim < 0) channel_dim += ndim;

        int64_t channel_size = input.size(channel_dim);
        int64_t batch_size = input.numel() / channel_size;

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_input = at::empty_like(input);

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != channel_dim) perm.push_back(i);
        }
        perm.push_back(channel_dim);

        at::Tensor permuted_in = input_contig.permute(perm).contiguous();
        at::Tensor permuted_grad_out = grad_output_contig.permute(perm).contiguous();
        at::Tensor permuted_grad_in = at::empty_like(permuted_in);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "identity_backward_cpu",
            [&]() {
                const scalar_t* grad_out_ptr = permuted_grad_out.data_ptr<scalar_t>();
                const scalar_t* in_ptr = permuted_in.data_ptr<scalar_t>();
                scalar_t* grad_in_ptr = permuted_grad_in.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        IdentityTraits::template backward_kernel<scalar_t>(
                            grad_out_ptr + b * channel_size,
                            in_ptr + b * channel_size,
                            grad_in_ptr + b * channel_size,
                            channel_size
                        );
                    }
                });
            }
        );

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
            inv_perm[perm[i]] = i;
        }

        grad_input = permuted_grad_in.permute(inv_perm).contiguous();
        return grad_input;
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t channel_dim = -1
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        int64_t ndim = input.dim();
        if (channel_dim < 0) channel_dim += ndim;

        int64_t channel_size = input.size(channel_dim);
        int64_t batch_size = input.numel() / channel_size;

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        at::Tensor grad_grad_output = at::empty_like(grad_output);
        at::Tensor new_grad_input = at::empty_like(input);

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != channel_dim) perm.push_back(i);
        }
        perm.push_back(channel_dim);

        at::Tensor permuted_in = input_contig.permute(perm).contiguous();
        at::Tensor permuted_grad_out = grad_output_contig.permute(perm).contiguous();
        at::Tensor permuted_gg_in = grad_grad_input_contig.permute(perm).contiguous();
        at::Tensor permuted_gg_out = at::empty_like(permuted_grad_out);
        at::Tensor permuted_new_grad = at::empty_like(permuted_in);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "identity_backward_backward_cpu",
            [&]() {
                const scalar_t* gg_in_ptr = permuted_gg_in.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = permuted_grad_out.data_ptr<scalar_t>();
                const scalar_t* in_ptr = permuted_in.data_ptr<scalar_t>();
                scalar_t* gg_out_ptr = permuted_gg_out.data_ptr<scalar_t>();
                scalar_t* new_grad_ptr = permuted_new_grad.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        IdentityTraits::template backward_backward_kernel<scalar_t>(
                            gg_in_ptr + b * channel_size,
                            grad_out_ptr + b * channel_size,
                            in_ptr + b * channel_size,
                            channel_size,
                            gg_out_ptr + b * channel_size,
                            new_grad_ptr + b * channel_size
                        );
                    }
                });
            }
        );

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
            inv_perm[perm[i]] = i;
        }

        grad_grad_output = permuted_gg_out.permute(inv_perm).contiguous();
        new_grad_input = permuted_new_grad.permute(inv_perm).contiguous();

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
};

}  // namespace torchscience::cpu
