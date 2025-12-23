#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu {

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

inline std::pair<int64_t, int64_t> compute_reduce_info(
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
        TORCH_CHECK(pos_d >= 0 && pos_d < ndim,
            "Dimension out of range (expected to be in range of [",
            -ndim, ", ", ndim - 1, "], but got ", d, ")");
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

inline std::vector<int64_t> compute_permutation(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    int64_t ndim = input.dim();
    std::vector<int64_t> permutation;

    if (!dim.has_value() || dim->empty()) {
        for (int64_t i = 0; i < ndim; ++i) {
            permutation.push_back(i);
        }
        return permutation;
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        TORCH_CHECK(pos_d >= 0 && pos_d < ndim,
            "Dimension out of range (expected to be in range of [",
            -ndim, ", ", ndim - 1, "], but got ", d, ")");
        reduce_dim[pos_d] = true;
    }

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

inline std::vector<int64_t> compute_inverse_permutation(
    const std::vector<int64_t>& permutation
) {
    int64_t ndim = permutation.size();
    std::vector<int64_t> inverse(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        inverse[permutation[i]] = i;
    }
    return inverse;
}

}  // namespace reduction_detail

// =============================================================================
// CPUReductionOperator - Template for reduction operators
// =============================================================================

// ReductionTraits must provide:
//   - template<T> static T reduce(const T* data, int64_t n, Args... args);
//   - template<T> static void backward(T grad, const T* data, int64_t n, T* grad_out, Args...);

template<typename ReductionTraits>
struct CPUReductionOperator {
    template<typename... Args>
    static at::Tensor forward(
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args... args
    ) {
        TORCH_CHECK(input.numel() > 0, "reduction: input tensor must be non-empty");

        auto output_shape = reduction_detail::compute_output_shape(input, dim, keepdim);
        auto [reduce_size, batch_size] = reduction_detail::compute_reduce_info(input, dim);

        at::Tensor input_contig = input.contiguous();
        auto options = input_contig.options();
        at::Tensor output = output_shape.empty()
            ? at::empty({}, options)
            : at::empty(output_shape, options);

        if (!dim.has_value() || dim->empty()) {
            // Reduce all dimensions
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_cpu_all",
                [&]() {
                    const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t result = ReductionTraits::template reduce<scalar_t>(
                        data_ptr, input_contig.numel(), args...
                    );
                    output.fill_(result);
                }
            );
        } else {
            // Dimension-specific reduction
            auto permutation = reduction_detail::compute_permutation(input, dim);
            at::Tensor permuted = input_contig.permute(permutation).contiguous();
            at::Tensor permuted_view = permuted.view({batch_size, reduce_size});

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_cpu_dim",
                [&]() {
                    const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                    scalar_t* output_ptr = output.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            output_ptr[b] = ReductionTraits::template reduce<scalar_t>(
                                data_ptr + b * reduce_size, reduce_size, args...
                            );
                        }
                    });
                }
            );
        }

        return output;
    }

    template<typename... Args>
    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args... args
    ) {
        at::Tensor grad_input = at::zeros_like(input);
        at::Tensor input_contig = input.contiguous();

        auto [reduce_size, batch_size] = reduction_detail::compute_reduce_info(input, dim);

        if (!dim.has_value() || dim->empty()) {
            // Scalar reduction
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_backward_cpu_all",
                [&]() {
                    scalar_t grad_out_val = grad_output.item<scalar_t>();
                    const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* grad_ptr = grad_input.data_ptr<scalar_t>();

                    ReductionTraits::template backward<scalar_t>(
                        grad_out_val, data_ptr, input_contig.numel(), grad_ptr, args...
                    );
                }
            );
        } else {
            // Dimension-specific backward
            auto permutation = reduction_detail::compute_permutation(input, dim);
            auto inverse_perm = reduction_detail::compute_inverse_permutation(permutation);

            at::Tensor permuted = input_contig.permute(permutation).contiguous();
            at::Tensor permuted_view = permuted.view({batch_size, reduce_size});
            at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, input.options());

            // Expand grad_output to match batch dimensions
            at::Tensor grad_output_expanded;
            if (keepdim) {
                grad_output_expanded = grad_output.contiguous().view({batch_size});
            } else {
                at::Tensor temp = grad_output;
                int64_t ndim = input.dim();
                std::vector<bool> reduce_dim(ndim, false);
                for (int64_t d : *dim) {
                    int64_t pos_d = d >= 0 ? d : d + ndim;
                    reduce_dim[pos_d] = true;
                }
                for (int64_t i = 0; i < ndim; ++i) {
                    if (reduce_dim[i]) {
                        temp = temp.unsqueeze(i);
                    }
                }
                grad_output_expanded = temp.contiguous().view({batch_size});
            }

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_backward_cpu_dim",
                [&]() {
                    const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                    const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
                    scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            ReductionTraits::template backward<scalar_t>(
                                grad_out_ptr[b],
                                data_ptr + b * reduce_size,
                                reduce_size,
                                grad_ptr + b * reduce_size,
                                args...
                            );
                        }
                    });
                }
            );

            grad_input = grad_permuted.view(permuted.sizes())
                .permute(inverse_perm)
                .contiguous();
        }

        return grad_input;
    }
};

}  // namespace torchscience::cpu
