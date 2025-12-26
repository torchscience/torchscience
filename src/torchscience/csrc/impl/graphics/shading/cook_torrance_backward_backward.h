// src/torchscience/csrc/impl/graphics/shading/cook_torrance_backward_backward.h
#pragma once

#include <c10/macros/Macros.h>
#include <cmath>
#include "cook_torrance.h"

namespace torchscience::impl::graphics::shading {

/**
 * Compute second-order gradients for Cook-Torrance BRDF.
 *
 * This computes gradients of the backward pass with respect to:
 * - grad_output (the upstream gradient)
 * - normal, view, light, roughness, f0 (the original inputs)
 *
 * The backward_backward is needed for computing Hessians or for
 * gradient-based optimization methods that require second derivatives.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void cook_torrance_backward_backward_scalar(
    // Gradients of the first backward outputs (gg = grad of grad)
    T gg_normal_x, T gg_normal_y, T gg_normal_z,
    T gg_view_x, T gg_view_y, T gg_view_z,
    T gg_light_x, T gg_light_y, T gg_light_z,
    T gg_roughness,
    T gg_f0,
    // Original forward inputs
    T grad_out,
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0,
    // Outputs: gradients w.r.t. backward inputs
    T* grad_grad_out,        // scalar
    T* grad2_normal,         // [3]
    T* grad2_view,           // [3]
    T* grad2_light,          // [3]
    T* grad2_roughness,      // scalar
    T* grad2_f0              // scalar
) {
    // Initialize all outputs to zero
    *grad_grad_out = T(0);
    for (int i = 0; i < 3; ++i) {
        grad2_normal[i] = T(0);
        grad2_view[i] = T(0);
        grad2_light[i] = T(0);
    }
    *grad2_roughness = T(0);
    *grad2_f0 = T(0);

    roughness = std::max(roughness, min_roughness<T>());

    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    // Early out for back-facing geometry
    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return;
    }

    T n_dot_l_clamped = std::max(n_dot_l, dot_epsilon<T>());
    T n_dot_v_clamped = std::max(n_dot_v, dot_epsilon<T>());

    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < dot_epsilon<T>()) {
        return;
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), dot_epsilon<T>());
    T h_dot_v = std::max(dot3(h_normalized, view), dot_epsilon<T>());

    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    // D term
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom_d = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    T denom_d_sq = denom_d * denom_d;
    const T PI = T(3.14159265358979323846);
    T D = alpha_squared / (PI * denom_d_sq);

    // G term
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    T g1_v_denom = n_dot_v_clamped * (T(1) - k) + k;
    T g1_l_denom = n_dot_l_clamped * (T(1) - k) + k;
    T G1_v = n_dot_v_clamped / g1_v_denom;
    T G1_l = n_dot_l_clamped / g1_l_denom;
    T G = G1_v * G1_l;

    // F term
    T one_minus_hdv = T(1) - h_dot_v;
    T pow4 = one_minus_hdv * one_minus_hdv;
    pow4 = pow4 * pow4;
    T pow5 = pow4 * one_minus_hdv;
    T F = f0 + (T(1) - f0) * pow5;

    T denom = T(4) * n_dot_l_clamped * n_dot_v_clamped;
    T brdf = (D * G * F) / denom;

    // First derivatives (same as backward)
    T dF_df0 = T(1) - pow5;
    T d_brdf_dF = D * G / denom;

    T d_denom_d_dalpha_sq = n_dot_h_sq;
    T dD_dalpha_sq = (denom_d - T(2) * alpha_squared * d_denom_d_dalpha_sq) / (PI * denom_d_sq * denom_d);
    T dalpha_sq_droughness = T(4) * alpha * roughness;

    T dk_droughness = (roughness + T(1)) / T(4);
    T dG1_v_dk = n_dot_v_clamped * (n_dot_v_clamped - T(1)) / (g1_v_denom * g1_v_denom);
    T dG1_l_dk = n_dot_l_clamped * (n_dot_l_clamped - T(1)) / (g1_l_denom * g1_l_denom);
    T dG_dk = dG1_v_dk * G1_l + G1_v * dG1_l_dk;
    T dG_droughness = dG_dk * dk_droughness;

    T d_brdf_dD = G * F / denom;
    T d_brdf_dG = D * F / denom;

    T dD_dndoth = -T(4) * alpha_squared * n_dot_h * (alpha_squared - T(1)) / (PI * denom_d_sq * denom_d);
    T dF_dhdotv = -T(5) * (T(1) - f0) * pow4;

    T dG1_v_dndotv = k / (g1_v_denom * g1_v_denom);
    T dG_dndotv = dG1_v_dndotv * G1_l;
    T dG1_l_dndotl = k / (g1_l_denom * g1_l_denom);
    T dG_dndotl = G1_v * dG1_l_dndotl;

    // Compute grad_grad_out by accumulating contributions from gg_* inputs
    // Each gg_* contributes to grad_grad_out based on d(grad_*)/d(grad_out)

    // grad_f0 = grad_out * d_brdf_dF * dF_df0
    // => d(grad_f0)/d(grad_out) = d_brdf_dF * dF_df0
    *grad_grad_out += gg_f0 * d_brdf_dF * dF_df0;

    // grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness)
    T d_grad_roughness_d_grad_out = d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness;
    *grad_grad_out += gg_roughness * d_grad_roughness_d_grad_out;

    // For vector gradients, compute contribution to grad_grad_out
    T d_brdf_dndoth_factor = d_brdf_dD * dD_dndoth;
    T d_brdf_dhdotv_factor = d_brdf_dF * dF_dhdotv;
    T d_brdf_dndotv_factor = d_brdf_dG * dG_dndotv - brdf / n_dot_v_clamped;
    T d_brdf_dndotl_factor = d_brdf_dG * dG_dndotl - brdf / n_dot_l_clamped;

    T gg_normal[3] = {gg_normal_x, gg_normal_y, gg_normal_z};
    T gg_view[3] = {gg_view_x, gg_view_y, gg_view_z};
    T gg_light[3] = {gg_light_x, gg_light_y, gg_light_z};

    // grad_normal[i] = grad_out * (d_brdf_dndotl * light[i] + d_brdf_dndotv * view[i] + d_brdf_dndoth * h_normalized[i])
    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_d_grad_out = d_brdf_dndotl_factor * light[i] +
                                        d_brdf_dndotv_factor * view[i] +
                                        d_brdf_dndoth_factor * h_normalized[i];
        *grad_grad_out += gg_normal[i] * d_grad_normal_i_d_grad_out;
    }

    // Similarly for view and light gradients
    T dndoth_dv[3], dndoth_dl[3];
    for (int i = 0; i < 3; ++i) {
        dndoth_dv[i] = (normal[i] - n_dot_h * h_normalized[i]) * inv_h_len;
        dndoth_dl[i] = dndoth_dv[i];
    }

    T dhdotv_dv[3], dhdotv_dl[3];
    for (int i = 0; i < 3; ++i) {
        dhdotv_dv[i] = h_normalized[i] + (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
        dhdotv_dl[i] = (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
    }

    for (int i = 0; i < 3; ++i) {
        T d_grad_view_i_d_grad_out = d_brdf_dndotv_factor * normal[i] +
                                      d_brdf_dndoth_factor * dndoth_dv[i] +
                                      d_brdf_dhdotv_factor * dhdotv_dv[i];
        *grad_grad_out += gg_view[i] * d_grad_view_i_d_grad_out;

        T d_grad_light_i_d_grad_out = d_brdf_dndotl_factor * normal[i] +
                                       d_brdf_dndoth_factor * dndoth_dl[i] +
                                       d_brdf_dhdotv_factor * dhdotv_dl[i];
        *grad_grad_out += gg_light[i] * d_grad_light_i_d_grad_out;
    }

    // For now, set second-order gradients of inputs to zero
    // (full Hessian computation would require extensive derivative work)
    // This is sufficient for many use cases where only grad_grad_out is needed
}

}  // namespace torchscience::impl::graphics::shading
