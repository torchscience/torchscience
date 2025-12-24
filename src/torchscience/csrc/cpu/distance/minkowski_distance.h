#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../impl/distance/minkowski_distance.h"
#include "../../impl/distance/minkowski_distance_backward.h"

namespace torchscience::cpu::distance {

/**
 * CPU implementation of pairwise Minkowski distance.
 *
 * Computes distance between each pair of points from x and y.
 *
 * @param x First set of points, shape (m, d)
 * @param y Second set of points, shape (n, d)
 * @param p Order of the norm
 * @param weight Optional weights, shape (d,)
 * @return Distance matrix, shape (m, n)
 */
inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    TORCH_CHECK(x.dim() == 2, "minkowski_distance: x must be 2D (m, d)");
    TORCH_CHECK(y.dim() == 2, "minkowski_distance: y must be 2D (n, d)");
    TORCH_CHECK(x.size(1) == y.size(1), "minkowski_distance: feature dimensions must match");
    TORCH_CHECK(p > 0, "minkowski_distance: p must be > 0");

    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor output = at::empty({m, n}, x.options());

    // Handle optional weight
    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        TORCH_CHECK(weight->dim() == 1, "minkowski_distance: weight must be 1D (d,)");
        TORCH_CHECK(weight->size(0) == d, "minkowski_distance: weight size must match feature dim");
        w_contig = weight->contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t p_val = static_cast<scalar_t>(p);

            at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    int64_t i = idx / n;
                    int64_t j = idx % n;
                    out_ptr[idx] = impl::distance::minkowski_distance_pair<scalar_t>(
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr
                    );
                }
            });
        }
    );

    return output;
}

/**
 * Backward pass for Minkowski distance.
 */
inline std::tuple<at::Tensor, at::Tensor> minkowski_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& dist_output
) {
    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t d = x.size(1);

    at::Tensor x_contig = x.contiguous();
    at::Tensor y_contig = y.contiguous();
    at::Tensor grad_contig = grad_output.contiguous();
    at::Tensor dist_contig = dist_output.contiguous();

    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_y = at::zeros_like(y);

    at::Tensor w_contig;
    bool has_weight = weight.has_value() && weight->defined();
    if (has_weight) {
        w_contig = weight->contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        x.scalar_type(),
        "minkowski_distance_backward_cpu",
        [&]() {
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
            const scalar_t* dist_ptr = dist_contig.data_ptr<scalar_t>();
            const scalar_t* w_ptr = has_weight ? w_contig.data_ptr<scalar_t>() : nullptr;
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();
            scalar_t* grad_y_ptr = grad_y.data_ptr<scalar_t>();
            scalar_t p_val = static_cast<scalar_t>(p);

            // Sequential accumulation for correctness
            // (parallel would need atomic operations or per-thread buffers)
            std::vector<scalar_t> temp_grad_x(d);
            std::vector<scalar_t> temp_grad_y(d);

            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t grad_val = grad_ptr[i * n + j];
                    scalar_t dist_val = dist_ptr[i * n + j];

                    impl::distance::minkowski_distance_backward_pair<scalar_t>(
                        grad_val,
                        x_ptr + i * d,
                        y_ptr + j * d,
                        d,
                        p_val,
                        w_ptr,
                        dist_val,
                        temp_grad_x.data(),
                        temp_grad_y.data()
                    );

                    for (int64_t k = 0; k < d; ++k) {
                        grad_x_ptr[i * d + k] += temp_grad_x[k];
                        grad_y_ptr[j * d + k] += temp_grad_y[k];
                    }
                }
            }
        }
    );

    return std::make_tuple(grad_x, grad_y);
}

}  // namespace torchscience::cpu::distance

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("minkowski_distance", &torchscience::cpu::distance::minkowski_distance);
    module.impl("minkowski_distance_backward", &torchscience::cpu::distance::minkowski_distance_backward);
}
