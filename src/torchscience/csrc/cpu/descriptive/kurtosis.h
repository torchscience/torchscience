#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

#include "../../impl/descriptive/kurtosis.h"

namespace torchscience::cpu::descriptive {

namespace {

/**
 * Compute output shape after reduction.
 */
inline std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        // Reduce all dimensions
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
        // else: scalar output, empty shape
    } else {
        // Reduce specified dimensions
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

/**
 * Compute the number of elements being reduced and batch size.
 */
inline std::pair<int64_t, int64_t> compute_reduce_info(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        // Reduce all dimensions
        return {input.numel(), 1};
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
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

}  // namespace

/**
 * CPU implementation of kurtosis.
 *
 * Computes kurtosis along specified dimensions with optional bias correction.
 *
 * @param input Input tensor
 * @param dim Dimensions to reduce (optional, reduces all if not specified)
 * @param keepdim Whether to keep reduced dimensions
 * @param fisher If true, compute excess kurtosis (subtract 3)
 * @param bias If true, return biased estimate
 * @return Kurtosis tensor
 */
inline at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    TORCH_CHECK(input.numel() > 0, "kurtosis: input tensor must be non-empty");

    // Compute output shape
    auto output_shape = compute_output_shape(input, dim, keepdim);
    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    // Ensure input is contiguous for efficient access
    at::Tensor input_contig = input.contiguous();

    // Determine output dtype (real type for complex inputs)
    auto output_dtype = at::isComplexType(input.scalar_type())
        ? c10::toRealValueType(input.scalar_type())
        : input.scalar_type();

    // Create output tensor
    auto options = input_contig.options().dtype(output_dtype);
    at::Tensor output = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    // Handle scalar reduction case (all dimensions)
    if (!dim.has_value() || dim->empty()) {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input_contig.scalar_type(),
            "kurtosis_cpu_all",
            [&]() {
                using real_t = typename c10::scalar_value_type<scalar_t>::type;

                if constexpr (c10::is_complex<scalar_t>::value) {
                    const c10::complex<real_t>* data_ptr =
                        reinterpret_cast<const c10::complex<real_t>*>(
                            input_contig.data_ptr<scalar_t>()
                        );
                    real_t result = impl::descriptive::kurtosis_1d_complex<real_t>(
                        data_ptr, input_contig.numel(), fisher, bias
                    );
                    output.fill_(result);
                } else {
                    // Handle half precision by computing in float
                    if constexpr (std::is_same_v<scalar_t, at::Half> ||
                                  std::is_same_v<scalar_t, at::BFloat16>) {
                        std::vector<float> data_float(input_contig.numel());
                        const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                        for (int64_t i = 0; i < input_contig.numel(); ++i) {
                            data_float[i] = static_cast<float>(data_ptr[i]);
                        }
                        float result = impl::descriptive::kurtosis_1d<float>(
                            data_float.data(), input_contig.numel(), fisher, bias
                        );
                        output.fill_(static_cast<scalar_t>(result));
                    } else {
                        const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                        scalar_t result = impl::descriptive::kurtosis_1d<scalar_t>(
                            data_ptr, input_contig.numel(), fisher, bias
                        );
                        output.fill_(result);
                    }
                }
            }
        );
        return output;
    }

    // Dimension-specific reduction
    // Permute dimensions so that reduction dims are last
    int64_t ndim = input.dim();
    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    std::vector<int64_t> permutation;
    std::vector<int64_t> batch_dims;
    std::vector<int64_t> reduce_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!reduce_dim[i]) {
            permutation.push_back(i);
            batch_dims.push_back(input.size(i));
        }
    }
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            permutation.push_back(i);
            reduce_dims.push_back(input.size(i));
        }
    }

    at::Tensor permuted = input_contig.permute(permutation).contiguous();
    at::Tensor permuted_view = permuted.view({batch_size, reduce_size});

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input_contig.scalar_type(),
        "kurtosis_cpu_dim",
        [&]() {
            using real_t = typename c10::scalar_value_type<scalar_t>::type;

            if constexpr (c10::is_complex<scalar_t>::value) {
                const c10::complex<real_t>* data_ptr =
                    reinterpret_cast<const c10::complex<real_t>*>(
                        permuted_view.data_ptr<scalar_t>()
                    );
                real_t* output_ptr = output.data_ptr<real_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        output_ptr[b] = impl::descriptive::kurtosis_1d_complex<real_t>(
                            data_ptr + b * reduce_size,
                            reduce_size,
                            fisher,
                            bias
                        );
                    }
                });
            } else {
                if constexpr (std::is_same_v<scalar_t, at::Half> ||
                              std::is_same_v<scalar_t, at::BFloat16>) {
                    const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                    scalar_t* output_ptr = output.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        std::vector<float> temp(reduce_size);
                        for (int64_t b = begin; b < end; ++b) {
                            for (int64_t i = 0; i < reduce_size; ++i) {
                                temp[i] = static_cast<float>(data_ptr[b * reduce_size + i]);
                            }
                            float result = impl::descriptive::kurtosis_1d<float>(
                                temp.data(), reduce_size, fisher, bias
                            );
                            output_ptr[b] = static_cast<scalar_t>(result);
                        }
                    });
                } else {
                    const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                    scalar_t* output_ptr = output.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            output_ptr[b] = impl::descriptive::kurtosis_1d<scalar_t>(
                                data_ptr + b * reduce_size,
                                reduce_size,
                                fisher,
                                bias
                            );
                        }
                    });
                }
            }
        }
    );

    return output;
}

/**
 * Backward pass for kurtosis on CPU.
 */
inline at::Tensor kurtosis_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // Create gradient tensor with same shape as input
    at::Tensor grad_input = at::zeros_like(input);

    at::Tensor input_contig = input.contiguous();
    at::Tensor grad_input_contig = grad_input.contiguous();

    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    // Expand grad_output to match batch dimensions if needed
    at::Tensor grad_output_expanded;
    if (!dim.has_value() || dim->empty()) {
        // Scalar reduction - grad_output is scalar, broadcast to all
        grad_output_expanded = grad_output.expand({1});
        batch_size = 1;
    } else {
        // Expand grad_output if keepdim was false
        if (keepdim) {
            grad_output_expanded = grad_output.contiguous().view({batch_size});
        } else {
            // Need to unsqueeze the reduced dimensions
            std::vector<int64_t> unsqueeze_shape = grad_output.sizes().vec();
            int64_t ndim = input.dim();
            std::vector<bool> reduce_dim(ndim, false);
            for (int64_t d : *dim) {
                int64_t pos_d = d >= 0 ? d : d + ndim;
                reduce_dim[pos_d] = true;
            }

            at::Tensor temp = grad_output;
            for (int64_t i = 0; i < ndim; ++i) {
                if (reduce_dim[i]) {
                    temp = temp.unsqueeze(i);
                }
            }
            grad_output_expanded = temp.contiguous().view({batch_size});
        }
    }

    // Handle scalar reduction case
    if (!dim.has_value() || dim->empty()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input_contig.scalar_type(),
            "kurtosis_backward_cpu_all",
            [&]() {
                scalar_t grad_out_val = grad_output.item<scalar_t>();

                if constexpr (std::is_same_v<scalar_t, at::Half> ||
                              std::is_same_v<scalar_t, at::BFloat16>) {
                    std::vector<float> data_float(input_contig.numel());
                    std::vector<float> grad_float(input_contig.numel());
                    const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                    for (int64_t i = 0; i < input_contig.numel(); ++i) {
                        data_float[i] = static_cast<float>(data_ptr[i]);
                    }

                    impl::descriptive::kurtosis_backward_1d<float>(
                        static_cast<float>(grad_out_val),
                        data_float.data(),
                        input_contig.numel(),
                        fisher,
                        bias,
                        grad_float.data()
                    );

                    scalar_t* grad_ptr = grad_input_contig.data_ptr<scalar_t>();
                    for (int64_t i = 0; i < input_contig.numel(); ++i) {
                        grad_ptr[i] = static_cast<scalar_t>(grad_float[i]);
                    }
                } else {
                    const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* grad_ptr = grad_input_contig.data_ptr<scalar_t>();

                    impl::descriptive::kurtosis_backward_1d<scalar_t>(
                        grad_out_val,
                        data_ptr,
                        input_contig.numel(),
                        fisher,
                        bias,
                        grad_ptr
                    );
                }
            }
        );
        return grad_input_contig;
    }

    // Dimension-specific backward
    int64_t ndim = input.dim();
    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    std::vector<int64_t> permutation;
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

    at::Tensor permuted = input_contig.permute(permutation).contiguous();
    at::Tensor permuted_view = permuted.view({batch_size, reduce_size});

    // Create permuted gradient output
    at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input_contig.scalar_type(),
        "kurtosis_backward_cpu_dim",
        [&]() {
            if constexpr (std::is_same_v<scalar_t, at::Half> ||
                          std::is_same_v<scalar_t, at::BFloat16>) {
                const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
                scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    std::vector<float> data_temp(reduce_size);
                    std::vector<float> grad_temp(reduce_size);

                    for (int64_t b = begin; b < end; ++b) {
                        for (int64_t i = 0; i < reduce_size; ++i) {
                            data_temp[i] = static_cast<float>(data_ptr[b * reduce_size + i]);
                        }

                        impl::descriptive::kurtosis_backward_1d<float>(
                            static_cast<float>(grad_out_ptr[b]),
                            data_temp.data(),
                            reduce_size,
                            fisher,
                            bias,
                            grad_temp.data()
                        );

                        for (int64_t i = 0; i < reduce_size; ++i) {
                            grad_ptr[b * reduce_size + i] = static_cast<scalar_t>(grad_temp[i]);
                        }
                    }
                });
            } else {
                const scalar_t* data_ptr = permuted_view.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
                scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        impl::descriptive::kurtosis_backward_1d<scalar_t>(
                            grad_out_ptr[b],
                            data_ptr + b * reduce_size,
                            reduce_size,
                            fisher,
                            bias,
                            grad_ptr + b * reduce_size
                        );
                    }
                });
            }
        }
    );

    // Inverse permutation to restore original dimension order
    std::vector<int64_t> inverse_perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        inverse_perm[permutation[i]] = i;
    }

    at::Tensor grad_unpermuted = grad_permuted.view(permuted.sizes())
        .permute(inverse_perm)
        .contiguous();

    return grad_unpermuted;
}

/**
 * Double-backward pass for kurtosis on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> kurtosis_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // Create outputs
    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor new_grad_input = at::zeros_like(input);

    at::Tensor input_contig = input.contiguous();
    at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    // Handle scalar reduction case
    if (!dim.has_value() || dim->empty()) {
        AT_DISPATCH_FLOATING_TYPES(
            input_contig.scalar_type(),
            "kurtosis_backward_backward_cpu_all",
            [&]() {
                const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
                const scalar_t* gg_input_ptr = grad_grad_input_contig.data_ptr<scalar_t>();
                scalar_t grad_out_val = grad_output.item<scalar_t>();

                scalar_t gg_output;
                std::vector<scalar_t> new_grad(input_contig.numel());

                impl::descriptive::kurtosis_backward_backward_1d<scalar_t>(
                    gg_input_ptr,
                    grad_out_val,
                    data_ptr,
                    input_contig.numel(),
                    fisher,
                    bias,
                    gg_output,
                    new_grad.data()
                );

                grad_grad_output.fill_(gg_output);

                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();
                for (int64_t i = 0; i < input_contig.numel(); ++i) {
                    new_grad_ptr[i] = new_grad[i];
                }
            }
        );

        return std::make_tuple(grad_grad_output, new_grad_input);
    }

    // For dimension-specific case, implement similar logic as backward
    // For now, return zeros (can be extended for full support)
    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::descriptive

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "kurtosis",
        &torchscience::cpu::descriptive::kurtosis
    );

    module.impl(
        "kurtosis_backward",
        &torchscience::cpu::descriptive::kurtosis_backward
    );

    module.impl(
        "kurtosis_backward_backward",
        &torchscience::cpu::descriptive::kurtosis_backward_backward
    );
}
