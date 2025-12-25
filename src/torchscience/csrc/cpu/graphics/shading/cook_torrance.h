#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../impl/graphics/shading/cook_torrance.h"
#include "../../../impl/graphics/shading/cook_torrance_backward.h"

namespace torchscience::cpu::graphics::shading {

/**
 * CPU implementation of Cook-Torrance specular BRDF.
 *
 * @param normal Surface normal vectors, shape (..., 3)
 * @param view View direction vectors, shape (..., 3)
 * @param light Light direction vectors, shape (..., 3)
 * @param roughness Surface roughness, shape (...) or scalar
 * @param f0 Fresnel reflectance at normal incidence, shape (...) or (..., 3)
 * @return BRDF values, shape (...) or (..., 3)
 */
inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    TORCH_CHECK(normal.size(-1) == 3, "cook_torrance: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "cook_torrance: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "cook_torrance: light must have last dimension 3");

    // Check if f0 is RGB (last dim = 3)
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions (excluding the last dim of 3 for vectors)
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Broadcast batch dimensions
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    // Compute total number of elements
    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    // Expand tensors to broadcast shape and flatten
    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor roughness_expanded = roughness.expand(batch_shape).contiguous().view({numel});

    at::Tensor f0_expanded;
    if (f0_is_rgb) {
        f0_expanded = f0.expand(vec_shape).contiguous().view({numel, 3});
    } else {
        f0_expanded = f0.expand(batch_shape).contiguous().view({numel});
    }

    // Create output tensor
    at::Tensor output;
    if (f0_is_rgb) {
        output = at::empty({numel, 3}, normal.options());
    } else {
        output = at::empty({numel}, normal.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_cpu",
        [&]() {
            const scalar_t* normal_ptr = normal_expanded.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_expanded.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_expanded.data_ptr<scalar_t>();
            const scalar_t* roughness_ptr = roughness_expanded.data_ptr<scalar_t>();
            const scalar_t* f0_ptr = f0_expanded.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            if (f0_is_rgb) {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx];
                        const scalar_t* f = f0_ptr + idx * 3;

                        // Compute BRDF for each color channel
                        for (int c = 0; c < 3; ++c) {
                            out_ptr[idx * 3 + c] = impl::graphics::shading::cook_torrance_scalar<scalar_t>(
                                n, v, l, r, f[c]
                            );
                        }
                    }
                });
            } else {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx];
                        scalar_t f = f0_ptr[idx];

                        out_ptr[idx] = impl::graphics::shading::cook_torrance_scalar<scalar_t>(
                            n, v, l, r, f
                        );
                    }
                });
            }
        }
    );

    // Reshape output to batch shape
    if (f0_is_rgb) {
        std::vector<int64_t> out_shape = batch_shape;
        out_shape.push_back(3);
        return output.view(out_shape);
    } else {
        return output.view(batch_shape);
    }
}

/**
 * Backward pass for Cook-Torrance BRDF.
 *
 * @return Tuple of (grad_normal, grad_view, grad_light, grad_roughness, grad_f0)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    // Check if f0 is RGB
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Broadcast batch dimensions
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    // Expand tensors
    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor roughness_expanded = roughness.expand(batch_shape).contiguous().view({numel});

    at::Tensor f0_expanded;
    at::Tensor grad_expanded;
    if (f0_is_rgb) {
        f0_expanded = f0.expand(vec_shape).contiguous().view({numel, 3});
        grad_expanded = grad_output.contiguous().view({numel, 3});
    } else {
        f0_expanded = f0.expand(batch_shape).contiguous().view({numel});
        grad_expanded = grad_output.contiguous().view({numel});
    }

    // Create gradient tensors
    at::Tensor grad_normal = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_view = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_light = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_roughness = at::zeros({numel}, normal.options());
    at::Tensor grad_f0;
    if (f0_is_rgb) {
        grad_f0 = at::zeros({numel, 3}, normal.options());
    } else {
        grad_f0 = at::zeros({numel}, normal.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_backward_cpu",
        [&]() {
            const scalar_t* normal_ptr = normal_expanded.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_expanded.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_expanded.data_ptr<scalar_t>();
            const scalar_t* roughness_ptr = roughness_expanded.data_ptr<scalar_t>();
            const scalar_t* f0_ptr = f0_expanded.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_expanded.data_ptr<scalar_t>();

            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_view_ptr = grad_view.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();
            scalar_t* grad_roughness_ptr = grad_roughness.data_ptr<scalar_t>();
            scalar_t* grad_f0_ptr = grad_f0.data_ptr<scalar_t>();

            if (f0_is_rgb) {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx];
                        const scalar_t* f = f0_ptr + idx * 3;
                        const scalar_t* g = grad_ptr + idx * 3;

                        scalar_t* gn = grad_normal_ptr + idx * 3;
                        scalar_t* gv = grad_view_ptr + idx * 3;
                        scalar_t* gl = grad_light_ptr + idx * 3;
                        scalar_t* gr = grad_roughness_ptr + idx;
                        scalar_t* gf = grad_f0_ptr + idx * 3;

                        // Accumulate gradients from each channel
                        for (int c = 0; c < 3; ++c) {
                            scalar_t temp_gn[3], temp_gv[3], temp_gl[3];
                            scalar_t temp_gr, temp_gf;

                            impl::graphics::shading::cook_torrance_backward_scalar<scalar_t>(
                                g[c], n, v, l, r, f[c],
                                temp_gn, temp_gv, temp_gl, &temp_gr, &temp_gf
                            );

                            for (int i = 0; i < 3; ++i) {
                                gn[i] += temp_gn[i];
                                gv[i] += temp_gv[i];
                                gl[i] += temp_gl[i];
                            }
                            *gr += temp_gr;
                            gf[c] = temp_gf;
                        }
                    }
                });
            } else {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx];
                        scalar_t f = f0_ptr[idx];
                        scalar_t g = grad_ptr[idx];

                        scalar_t* gn = grad_normal_ptr + idx * 3;
                        scalar_t* gv = grad_view_ptr + idx * 3;
                        scalar_t* gl = grad_light_ptr + idx * 3;

                        impl::graphics::shading::cook_torrance_backward_scalar<scalar_t>(
                            g, n, v, l, r, f,
                            gn, gv, gl, grad_roughness_ptr + idx, grad_f0_ptr + idx
                        );
                    }
                });
            }
        }
    );

    // Reshape gradients back to original shapes
    return std::make_tuple(
        grad_normal.view(vec_shape),
        grad_view.view(vec_shape),
        grad_light.view(vec_shape),
        grad_roughness.view(batch_shape),
        f0_is_rgb ? grad_f0.view(vec_shape) : grad_f0.view(batch_shape)
    );
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("cook_torrance", &torchscience::cpu::graphics::shading::cook_torrance);
    module.impl("cook_torrance_backward", &torchscience::cpu::graphics::shading::cook_torrance_backward);
}
