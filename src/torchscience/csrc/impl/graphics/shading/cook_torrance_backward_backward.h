// src/torchscience/csrc/impl/graphics/shading/cook_torrance_backward_backward.h
#pragma once

/*
 * Cook-Torrance BRDF Second-Order Gradients (Hessian)
 *
 * This file implements the backward_backward pass for Cook-Torrance BRDF,
 * computing full second-order derivatives (Hessian) for all inputs.
 *
 * The backward function computes:
 *   grad_f0 = grad_out * d_brdf_dF * dF_df0
 *   grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_dr + d_brdf_dG * dG_dr)
 *   grad_normal[i] = grad_out * (d_brdf_dndotl * l[i] + d_brdf_dndotv * v[i] + d_brdf_dndoth * h[i])
 *   grad_view[i] = grad_out * (d_brdf_dndotv * n[i] + d_brdf_dndoth * dndoth_dv[i] + d_brdf_dhdotv * dhdotv_dv[i])
 *   grad_light[i] = grad_out * (d_brdf_dndotl * n[i] + d_brdf_dndoth * dndoth_dl[i] + d_brdf_dhdotv * dhdotv_dl[i])
 *
 * For backward_backward, we compute:
 *   grad2_X = sum over outputs O: gg_O * d(grad_O)/d(X)
 */

#include <c10/macros/Macros.h>
#include <algorithm>
#include <cmath>
#include "cook_torrance.h"

namespace torchscience::impl::graphics::shading {

/**
 * Compute second-order gradients for Cook-Torrance BRDF.
 *
 * This computes the full Hessian, providing gradients of the backward pass
 * outputs with respect to all original inputs.
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

    roughness = std::clamp(roughness, min_roughness<T>(), T(1));

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
    T inv_h_len_sq = inv_h_len * inv_h_len;
    T inv_h_len_cu = inv_h_len_sq * inv_h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), dot_epsilon<T>());
    T h_dot_v = std::max(dot3(h_normalized, view), dot_epsilon<T>());

    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    // =========================================================================
    // Forward pass values (needed for derivatives)
    // =========================================================================

    // D term
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom_d = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    T denom_d_sq = denom_d * denom_d;
    T denom_d_cu = denom_d_sq * denom_d;
    const T PI = T(3.14159265358979323846);
    T D = alpha_squared / (PI * denom_d_sq);

    // G term
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    T one_minus_k = T(1) - k;
    T g1_v_denom = n_dot_v_clamped * one_minus_k + k;
    T g1_l_denom = n_dot_l_clamped * one_minus_k + k;
    T g1_v_denom_sq = g1_v_denom * g1_v_denom;
    T g1_l_denom_sq = g1_l_denom * g1_l_denom;
    T G1_v = n_dot_v_clamped / g1_v_denom;
    T G1_l = n_dot_l_clamped / g1_l_denom;
    T G = G1_v * G1_l;

    // F term
    T one_minus_hdv = T(1) - h_dot_v;
    T pow2 = one_minus_hdv * one_minus_hdv;
    T pow3 = pow2 * one_minus_hdv;
    T pow4 = pow2 * pow2;
    T pow5 = pow4 * one_minus_hdv;
    T one_minus_f0 = T(1) - f0;
    T F = f0 + one_minus_f0 * pow5;

    T denom = T(4) * n_dot_l_clamped * n_dot_v_clamped;
    T inv_denom = T(1) / denom;
    T brdf = (D * G * F) * inv_denom;

    // =========================================================================
    // First derivatives (from backward pass)
    // =========================================================================

    // Derivatives of F
    T dF_df0 = T(1) - pow5;
    T dF_dhdotv = -T(5) * one_minus_f0 * pow4;

    // Derivatives of D w.r.t. n_dot_h and alpha_squared
    T dD_dndoth = -T(4) * alpha_squared * n_dot_h * (alpha_squared - T(1)) / (PI * denom_d_cu);
    T dD_dalpha_sq = (denom_d - T(2) * alpha_squared * n_dot_h_sq) / (PI * denom_d_cu);

    // Derivatives of alpha_squared w.r.t. roughness
    T dalpha_droughness = T(2) * roughness;
    T dalpha_sq_droughness = T(4) * alpha * roughness;

    // Derivatives of G w.r.t. n_dot_v, n_dot_l, and k
    T dG1_v_dndotv = k / g1_v_denom_sq;
    T dG1_l_dndotl = k / g1_l_denom_sq;
    T dG1_v_dk = n_dot_v_clamped * (n_dot_v_clamped - T(1)) / g1_v_denom_sq;
    T dG1_l_dk = n_dot_l_clamped * (n_dot_l_clamped - T(1)) / g1_l_denom_sq;

    // Derivatives of k w.r.t. roughness
    T dk_droughness = r_plus_1 / T(4);

    // Derivatives of G w.r.t. inputs
    T dG_dndotv = dG1_v_dndotv * G1_l;
    T dG_dndotl = G1_v * dG1_l_dndotl;
    T dG_dk = dG1_v_dk * G1_l + G1_v * dG1_l_dk;
    T dG_droughness = dG_dk * dk_droughness;

    // Composite derivatives of brdf
    T d_brdf_dF = D * G * inv_denom;
    T d_brdf_dD = G * F * inv_denom;
    T d_brdf_dG = D * F * inv_denom;

    T d_brdf_dndotv = d_brdf_dG * dG_dndotv - brdf / n_dot_v_clamped;
    T d_brdf_dndotl = d_brdf_dG * dG_dndotl - brdf / n_dot_l_clamped;
    T d_brdf_dndoth = d_brdf_dD * dD_dndoth;
    T d_brdf_dhdotv = d_brdf_dF * dF_dhdotv;

    // Derivatives of h_normalized w.r.t. view and light
    // h = v + l, h_normalized = h / |h|
    // dh_normalized/dv = (I - h_normalized ⊗ h_normalized) / |h|
    T dndoth_dv[3], dndoth_dl[3];
    T dhdotv_dv[3], dhdotv_dl[3];
    for (int i = 0; i < 3; ++i) {
        // d(n·h)/dv = d(n·(h/|h|))/dv = n · d(h/|h|)/dv
        // = n · (I/|h| - h⊗h/|h|³) = (n - (n·h)·h_normalized) / |h|
        dndoth_dv[i] = (normal[i] - n_dot_h * h_normalized[i]) * inv_h_len;
        dndoth_dl[i] = dndoth_dv[i];  // Same formula

        // d(h·v)/dv = d((h/|h|)·v)/dv = h_normalized + v·d(h/|h|)/dv
        // = h_normalized + (v - (h·v)·h_normalized) / |h|
        dhdotv_dv[i] = h_normalized[i] + (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
        dhdotv_dl[i] = (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
    }

    // =========================================================================
    // Collect gg vectors for convenience
    // =========================================================================
    T gg_normal[3] = {gg_normal_x, gg_normal_y, gg_normal_z};
    T gg_view[3] = {gg_view_x, gg_view_y, gg_view_z};
    T gg_light[3] = {gg_light_x, gg_light_y, gg_light_z};

    // =========================================================================
    // grad_grad_out: gradient w.r.t. grad_out
    // This is d(grad_X)/d(grad_out) = (coefficient of grad_out in grad_X)
    // Already computed in original implementation - keep it
    // =========================================================================

    // grad_f0 = grad_out * d_brdf_dF * dF_df0
    // => d(grad_f0)/d(grad_out) = d_brdf_dF * dF_df0
    *grad_grad_out += gg_f0 * d_brdf_dF * dF_df0;

    // grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_dr + d_brdf_dG * dG_dr)
    T d_grad_roughness_d_grad_out = d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness;
    *grad_grad_out += gg_roughness * d_grad_roughness_d_grad_out;

    // grad_normal[i] = grad_out * (d_brdf_dndotl * l[i] + d_brdf_dndotv * v[i] + d_brdf_dndoth * h[i])
    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_d_grad_out = d_brdf_dndotl * light[i] +
                                        d_brdf_dndotv * view[i] +
                                        d_brdf_dndoth * h_normalized[i];
        *grad_grad_out += gg_normal[i] * d_grad_normal_i_d_grad_out;
    }

    // grad_view[i] = grad_out * (d_brdf_dndotv * n[i] + d_brdf_dndoth * dndoth_dv[i] + d_brdf_dhdotv * dhdotv_dv[i])
    for (int i = 0; i < 3; ++i) {
        T d_grad_view_i_d_grad_out = d_brdf_dndotv * normal[i] +
                                      d_brdf_dndoth * dndoth_dv[i] +
                                      d_brdf_dhdotv * dhdotv_dv[i];
        *grad_grad_out += gg_view[i] * d_grad_view_i_d_grad_out;
    }

    // grad_light[i] = grad_out * (d_brdf_dndotl * n[i] + d_brdf_dndoth * dndoth_dl[i] + d_brdf_dhdotv * dhdotv_dl[i])
    for (int i = 0; i < 3; ++i) {
        T d_grad_light_i_d_grad_out = d_brdf_dndotl * normal[i] +
                                       d_brdf_dndoth * dndoth_dl[i] +
                                       d_brdf_dhdotv * dhdotv_dl[i];
        *grad_grad_out += gg_light[i] * d_grad_light_i_d_grad_out;
    }

    // =========================================================================
    // Second derivatives of intermediate quantities
    // These are needed for computing grad2_* below
    // =========================================================================

    // Second derivative of D w.r.t. n_dot_h
    // dD_dndoth = -4 * alpha_sq * n_dot_h * (alpha_sq - 1) / (PI * denom_d^3)
    // d²D/d(n_dot_h)² = d/d(n_dot_h)[-4 * alpha_sq * n_dot_h * (alpha_sq - 1) / (PI * denom_d^3)]
    T d2D_dndoth2;
    {
        // Let A = -4 * alpha_sq * (alpha_sq - 1) / PI
        // dD_dndoth = A * n_dot_h / denom_d^3
        // d²D/d(n_dot_h)² = A * d/d(n_dot_h)[n_dot_h / denom_d^3]
        // = A * (1 / denom_d^3 - n_dot_h * 3 * denom_d^2 * d(denom_d)/d(n_dot_h) / denom_d^6)
        // d(denom_d)/d(n_dot_h) = 2 * n_dot_h * (alpha_sq - 1)
        // = A * (1/denom_d^3 - 6 * n_dot_h^2 * (alpha_sq - 1) / denom_d^4)
        // = A / denom_d^3 * (1 - 6 * n_dot_h^2 * (alpha_sq - 1) / denom_d)
        T A = -T(4) * alpha_squared * (alpha_squared - T(1)) / PI;
        T factor = T(1) - T(6) * n_dot_h_sq * (alpha_squared - T(1)) / denom_d;
        d2D_dndoth2 = A / denom_d_cu * factor;
    }

    // Second derivative of F w.r.t. h_dot_v
    // dF_dhdotv = -5 * (1 - f0) * pow4
    // d²F/d(h_dot_v)² = -5 * (1 - f0) * 4 * pow3 * (-1) = 20 * (1 - f0) * pow3
    T d2F_dhdotv2 = T(20) * one_minus_f0 * pow3;

    // Cross derivative of D w.r.t. n_dot_h and alpha_squared
    // dD_dndoth = -4 * alpha_sq * n_dot_h * (alpha_sq - 1) / (PI * denom_d^3)
    // d²D/(d(n_dot_h) d(alpha_sq))
    T d2D_dndoth_dalpha_sq;
    {
        // This is complex. Let's use the product rule carefully.
        // dD_dndoth = C * n_dot_h / denom_d^3  where C = -4 * alpha_sq * (alpha_sq - 1) / PI
        // d/d(alpha_sq)[C] = -4 * (2*alpha_sq - 1) / PI
        // d/d(alpha_sq)[1/denom_d^3] = -3 / denom_d^4 * n_dot_h^2
        // d²D/(d(n_dot_h) d(alpha_sq)) = d/d(alpha_sq)[C] * n_dot_h / denom_d^3
        //                              + C * n_dot_h * d/d(alpha_sq)[1/denom_d^3]
        T dC_dalpha_sq = -T(4) * (T(2) * alpha_squared - T(1)) / PI;
        T C = -T(4) * alpha_squared * (alpha_squared - T(1)) / PI;
        T d_inv_denom_d_cu_dalpha_sq = -T(3) / (denom_d_sq * denom_d_sq) * n_dot_h_sq;
        d2D_dndoth_dalpha_sq = dC_dalpha_sq * n_dot_h / denom_d_cu + C * n_dot_h * d_inv_denom_d_cu_dalpha_sq;
    }

    // Second derivative of D w.r.t. alpha_squared
    // dD_dalpha_sq = (denom_d - 2 * alpha_sq * n_dot_h^2) / (PI * denom_d^3)
    T d2D_dalpha_sq2;
    {
        // Let num = denom_d - 2 * alpha_sq * n_dot_h^2
        // dD_dalpha_sq = num / (PI * denom_d^3)
        // d(num)/d(alpha_sq) = n_dot_h^2 - 2 * n_dot_h^2 = -n_dot_h^2
        // d(denom_d^(-3))/d(alpha_sq) = -3 * denom_d^(-4) * n_dot_h^2
        // d²D/d(alpha_sq)² = (-n_dot_h^2) / (PI * denom_d^3) + num * (-3) / (PI * denom_d^4) * n_dot_h^2
        T num = denom_d - T(2) * alpha_squared * n_dot_h_sq;
        d2D_dalpha_sq2 = (-n_dot_h_sq) / (PI * denom_d_cu) + num * (-T(3)) / (PI * denom_d_sq * denom_d_sq) * n_dot_h_sq;
    }

    // Second derivatives of G1 terms
    // dG1_v_dndotv = k / g1_v_denom^2
    // d²G1_v/d(n_dot_v)² = k * (-2) / g1_v_denom^3 * (1 - k) = -2 * k * (1 - k) / g1_v_denom^3
    T d2G1_v_dndotv2 = -T(2) * k * one_minus_k / (g1_v_denom_sq * g1_v_denom);
    T d2G1_l_dndotl2 = -T(2) * k * one_minus_k / (g1_l_denom_sq * g1_l_denom);

    // Cross derivatives of G1 w.r.t. n_dot and k
    // dG1_v_dndotv = k / g1_v_denom^2
    // d²G1_v/(d(n_dot_v) d(k)) = 1/g1_v_denom^2 + k * (-2) / g1_v_denom^3 * (1 - n_dot_v)
    //                         = (1/g1_v_denom^2) * (1 - 2k(1 - n_dot_v)/g1_v_denom)
    T d2G1_v_dndotv_dk = T(1)/g1_v_denom_sq + k * (-T(2)) / (g1_v_denom_sq * g1_v_denom) * (T(1) - n_dot_v_clamped);
    T d2G1_l_dndotl_dk = T(1)/g1_l_denom_sq + k * (-T(2)) / (g1_l_denom_sq * g1_l_denom) * (T(1) - n_dot_l_clamped);

    // Second derivative of G1_v w.r.t. k
    // dG1_v_dk = n_dot_v * (n_dot_v - 1) / g1_v_denom^2
    // d²G1_v/dk² = n_dot_v * (n_dot_v - 1) * (-2) / g1_v_denom^3 * (1 - n_dot_v)
    //            = 2 * n_dot_v * (1 - n_dot_v)^2 / g1_v_denom^3
    T d2G1_v_dk2 = T(2) * n_dot_v_clamped * (T(1) - n_dot_v_clamped) * (T(1) - n_dot_v_clamped) / (g1_v_denom_sq * g1_v_denom);
    T d2G1_l_dk2 = T(2) * n_dot_l_clamped * (T(1) - n_dot_l_clamped) * (T(1) - n_dot_l_clamped) / (g1_l_denom_sq * g1_l_denom);

    // Second derivative of k w.r.t. roughness
    T d2k_droughness2 = T(1) / T(4);

    // Second derivative of alpha_squared w.r.t. roughness
    // dalpha_sq_droughness = 4 * alpha * roughness = 4 * r^3
    // d²(alpha_sq)/dr² = 12 * r^2
    T d2alpha_sq_droughness2 = T(12) * roughness * roughness;

    // =========================================================================
    // Second derivatives of composite brdf terms
    // =========================================================================

    // d_brdf_dF = D * G / denom
    // d(d_brdf_dF)/dF = 0 (F doesn't appear in d_brdf_dF)
    // d(d_brdf_dF)/dD = G / denom
    // d(d_brdf_dF)/dG = D / denom
    // d(d_brdf_dF)/d(denom) = -D * G / denom^2
    T d_d_brdf_dF_dD = G * inv_denom;
    T d_d_brdf_dF_dG = D * inv_denom;
    T d_d_brdf_dF_d_inv_denom = D * G;

    // d_brdf_dD = G * F / denom
    T d_d_brdf_dD_dF = G * inv_denom;
    T d_d_brdf_dD_dG = F * inv_denom;
    T d_d_brdf_dD_d_inv_denom = G * F;

    // d_brdf_dG = D * F / denom
    T d_d_brdf_dG_dF = D * inv_denom;
    T d_d_brdf_dG_dD = F * inv_denom;
    T d_d_brdf_dG_d_inv_denom = D * F;

    // d(inv_denom)/d(n_dot_l) = -inv_denom / n_dot_l
    // d(inv_denom)/d(n_dot_v) = -inv_denom / n_dot_v
    T d_inv_denom_dndotl = -inv_denom / n_dot_l_clamped;
    T d_inv_denom_dndotv = -inv_denom / n_dot_v_clamped;

    // =========================================================================
    // grad2_f0: gradient w.r.t. f0
    // =========================================================================

    // Contributions from gg_f0:
    // grad_f0 = grad_out * d_brdf_dF * dF_df0
    // d(grad_f0)/df0:
    // - dF_df0 = 1 - pow5, d(dF_df0)/df0 = 0
    // - d_brdf_dF = D * G / denom, d(d_brdf_dF)/df0 = 0
    // => d(grad_f0)/df0 = 0

    // Contributions from gg_roughness:
    // grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_dr + d_brdf_dG * dG_dr)
    // d_brdf_dD = G * F / denom, d(d_brdf_dD)/df0 = G / denom * dF/df0 = G / denom * (1 - pow5)
    // d_brdf_dG = D * F / denom, d(d_brdf_dG)/df0 = D / denom * dF/df0 = D / denom * (1 - pow5)
    T d_d_brdf_dD_df0 = d_d_brdf_dD_dF * dF_df0;
    T d_d_brdf_dG_df0 = d_d_brdf_dG_dF * dF_df0;
    T d_grad_roughness_df0 = grad_out * (d_d_brdf_dD_df0 * dD_dalpha_sq * dalpha_sq_droughness +
                                          d_d_brdf_dG_df0 * dG_droughness);
    *grad2_f0 += gg_roughness * d_grad_roughness_df0;

    // Contributions from gg_normal:
    // grad_normal[i] = grad_out * (d_brdf_dndotl * l[i] + d_brdf_dndotv * v[i] + d_brdf_dndoth * h[i])
    // d_brdf_dndotl = d_brdf_dG * dG_dndotl - brdf / n_dot_l
    // d_brdf_dndotv = d_brdf_dG * dG_dndotv - brdf / n_dot_v
    // d_brdf_dndoth = d_brdf_dD * dD_dndoth
    // d_brdf_dhdotv = d_brdf_dF * dF_dhdotv

    // d(d_brdf_dndotl)/df0 = d(d_brdf_dG)/df0 * dG_dndotl - d(brdf)/df0 / n_dot_l
    // d(brdf)/df0 = d_brdf_dF * dF_df0
    T d_brdf_df0 = d_brdf_dF * dF_df0;
    T d_d_brdf_dndotl_df0 = d_d_brdf_dG_df0 * dG_dndotl - d_brdf_df0 / n_dot_l_clamped;
    T d_d_brdf_dndotv_df0 = d_d_brdf_dG_df0 * dG_dndotv - d_brdf_df0 / n_dot_v_clamped;
    T d_d_brdf_dndoth_df0 = d_d_brdf_dD_df0 * dD_dndoth;
    T d_d_brdf_dhdotv_df0 = d_d_brdf_dF_dD * T(0) * dF_dhdotv + d_brdf_dF * T(0);  // dF_dhdotv = -5*(1-f0)*pow4
    // Actually dF_dhdotv = -5 * (1-f0) * pow4, so d(dF_dhdotv)/df0 = 5 * pow4
    T d_dF_dhdotv_df0 = T(5) * pow4;
    d_d_brdf_dhdotv_df0 = d_d_brdf_dF_dD * T(0) * dF_dhdotv + d_brdf_dF * d_dF_dhdotv_df0;  // d_brdf_dF doesn't depend on f0 directly here
    // Wait, d_brdf_dF = D*G/denom which doesn't depend on f0, so d(d_brdf_dF)/df0 = 0
    d_d_brdf_dhdotv_df0 = d_brdf_dF * d_dF_dhdotv_df0;

    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_df0 = grad_out * (d_d_brdf_dndotl_df0 * light[i] +
                                             d_d_brdf_dndotv_df0 * view[i] +
                                             d_d_brdf_dndoth_df0 * h_normalized[i]);
        *grad2_f0 += gg_normal[i] * d_grad_normal_i_df0;

        T d_grad_view_i_df0 = grad_out * (d_d_brdf_dndotv_df0 * normal[i] +
                                           d_d_brdf_dndoth_df0 * dndoth_dv[i] +
                                           d_d_brdf_dhdotv_df0 * dhdotv_dv[i]);
        *grad2_f0 += gg_view[i] * d_grad_view_i_df0;

        T d_grad_light_i_df0 = grad_out * (d_d_brdf_dndotl_df0 * normal[i] +
                                            d_d_brdf_dndoth_df0 * dndoth_dl[i] +
                                            d_d_brdf_dhdotv_df0 * dhdotv_dl[i]);
        *grad2_f0 += gg_light[i] * d_grad_light_i_df0;
    }

    // =========================================================================
    // grad2_roughness: gradient w.r.t. roughness
    // =========================================================================

    // This requires computing d(grad_*)/d(roughness) for all gradient outputs

    // Intermediate derivatives w.r.t. roughness:
    // d(alpha_sq)/dr = dalpha_sq_droughness
    // d(D)/dr = dD_dalpha_sq * dalpha_sq_droughness
    // d(G)/dr = dG_droughness
    // d(k)/dr = dk_droughness
    // d(denom)/dr = 0 (denom = 4 * n_dot_l * n_dot_v)
    // d(brdf)/dr = (dD/dr * G * F + D * dG/dr * F) / denom
    //            = (dD_dalpha_sq * dalpha_sq_dr * G * F + D * dG_dr * F) / denom
    T dD_droughness = dD_dalpha_sq * dalpha_sq_droughness;
    T d_brdf_droughness = (dD_droughness * G * F + D * dG_droughness * F) * inv_denom;

    // d(d_brdf_dD)/dr = d(G * F / denom)/dr = dG/dr * F / denom = dG_droughness * F * inv_denom
    T d_d_brdf_dD_dr = dG_droughness * F * inv_denom;
    // d(d_brdf_dG)/dr = d(D * F / denom)/dr = dD/dr * F / denom = dD_droughness * F * inv_denom
    T d_d_brdf_dG_dr = dD_droughness * F * inv_denom;
    // d(d_brdf_dF)/dr = d(D * G / denom)/dr = (dD/dr * G + D * dG/dr) / denom
    T d_d_brdf_dF_dr = (dD_droughness * G + D * dG_droughness) * inv_denom;

    // d(dD_dalpha_sq)/dr = d²D/d(alpha_sq)² * dalpha_sq_dr
    T d_dD_dalpha_sq_dr = d2D_dalpha_sq2 * dalpha_sq_droughness;
    // d(dalpha_sq_dr)/dr = d²(alpha_sq)/dr² = 12 * r²
    T d_dalpha_sq_dr_dr = d2alpha_sq_droughness2;

    // d(dG_droughness)/dr = d/dr[(dG1_v_dk * G1_l + G1_v * dG1_l_dk) * dk_dr]
    // This is complex. Let's compute step by step.
    // dG1_v_dk = n_dot_v * (n_dot_v - 1) / g1_v_denom^2
    // d(dG1_v_dk)/dr = d(dG1_v_dk)/dk * dk_dr
    // d(dG1_v_dk)/dk = d²G1_v/dk² = 2 * n_dot_v * (1 - n_dot_v)^2 / g1_v_denom^3
    T d_dG1_v_dk_dr = d2G1_v_dk2 * dk_droughness;
    T d_dG1_l_dk_dr = d2G1_l_dk2 * dk_droughness;
    // d(G1_v)/dr = dG1_v_dk * dk_dr
    T dG1_v_dr = dG1_v_dk * dk_droughness;
    T dG1_l_dr = dG1_l_dk * dk_droughness;
    // d(dk_dr)/dr = d²k/dr² = 1/4
    T d_dk_dr_dr = d2k_droughness2;

    // dG_dr = (dG1_v_dk * G1_l + G1_v * dG1_l_dk) * dk_dr
    // d(dG_dr)/dr = (d(dG1_v_dk)/dr * G1_l + dG1_v_dk * dG1_l/dr +
    //               dG1_v/dr * dG1_l_dk + G1_v * d(dG1_l_dk)/dr) * dk_dr
    //            + (dG1_v_dk * G1_l + G1_v * dG1_l_dk) * d(dk_dr)/dr
    T d_dG_dr_dr = (d_dG1_v_dk_dr * G1_l + dG1_v_dk * dG1_l_dr +
                    dG1_v_dr * dG1_l_dk + G1_v * d_dG1_l_dk_dr) * dk_droughness +
                   (dG1_v_dk * G1_l + G1_v * dG1_l_dk) * d_dk_dr_dr;

    // d(dD_dndoth)/dr = d²D/(d(n_dot_h) d(alpha_sq)) * dalpha_sq_dr
    T d_dD_dndoth_dr = d2D_dndoth_dalpha_sq * dalpha_sq_droughness;

    // d(d_brdf_dndotl)/dr = d(d_brdf_dG * dG_dndotl - brdf / n_dot_l)/dr
    //                     = d(d_brdf_dG)/dr * dG_dndotl - d(brdf)/dr / n_dot_l
    // (dG_dndotl doesn't depend on roughness since it depends on k through g1_l_denom)
    // Actually dG_dndotl = G1_v * dG1_l_dndotl where dG1_l_dndotl = k / g1_l_denom^2
    // d(dG_dndotl)/dr = dG1_v_dr * dG1_l_dndotl + G1_v * d(dG1_l_dndotl)/dr
    // d(dG1_l_dndotl)/dr = d(dG1_l_dndotl)/dk * dk_dr = d²G1_l/(d(n_dot_l) dk) * dk_dr
    T d_dG1_l_dndotl_dr = d2G1_l_dndotl_dk * dk_droughness;
    T d_dG1_v_dndotv_dr = d2G1_v_dndotv_dk * dk_droughness;
    T d_dG_dndotl_dr = dG1_v_dr * dG1_l_dndotl + G1_v * d_dG1_l_dndotl_dr;
    T d_dG_dndotv_dr = d_dG1_v_dndotv_dr * G1_l + dG1_v_dndotv * dG1_l_dr;

    T d_d_brdf_dndotl_dr = d_d_brdf_dG_dr * dG_dndotl + d_brdf_dG * d_dG_dndotl_dr - d_brdf_droughness / n_dot_l_clamped;
    T d_d_brdf_dndotv_dr = d_d_brdf_dG_dr * dG_dndotv + d_brdf_dG * d_dG_dndotv_dr - d_brdf_droughness / n_dot_v_clamped;
    T d_d_brdf_dndoth_dr = d_d_brdf_dD_dr * dD_dndoth + d_brdf_dD * d_dD_dndoth_dr;

    // Contributions from gg_roughness:
    // grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_dr + d_brdf_dG * dG_dr)
    // d(grad_roughness)/dr = grad_out * (d(d_brdf_dD)/dr * dD_dalpha_sq * dalpha_sq_dr +
    //                                    d_brdf_dD * d(dD_dalpha_sq)/dr * dalpha_sq_dr +
    //                                    d_brdf_dD * dD_dalpha_sq * d(dalpha_sq_dr)/dr +
    //                                    d(d_brdf_dG)/dr * dG_dr +
    //                                    d_brdf_dG * d(dG_dr)/dr)
    T d_grad_roughness_dr = grad_out * (d_d_brdf_dD_dr * dD_dalpha_sq * dalpha_sq_droughness +
                                         d_brdf_dD * d_dD_dalpha_sq_dr * dalpha_sq_droughness +
                                         d_brdf_dD * dD_dalpha_sq * d_dalpha_sq_dr_dr +
                                         d_d_brdf_dG_dr * dG_droughness +
                                         d_brdf_dG * d_dG_dr_dr);
    *grad2_roughness += gg_roughness * d_grad_roughness_dr;

    // Contributions from gg_f0:
    // grad_f0 = grad_out * d_brdf_dF * dF_df0
    // d(grad_f0)/dr = grad_out * d(d_brdf_dF)/dr * dF_df0
    T d_grad_f0_dr = grad_out * d_d_brdf_dF_dr * dF_df0;
    *grad2_roughness += gg_f0 * d_grad_f0_dr;

    // Contributions from gg_normal, gg_view, gg_light:
    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_dr = grad_out * (d_d_brdf_dndotl_dr * light[i] +
                                            d_d_brdf_dndotv_dr * view[i] +
                                            d_d_brdf_dndoth_dr * h_normalized[i]);
        *grad2_roughness += gg_normal[i] * d_grad_normal_i_dr;

        // d_brdf_dhdotv = d_brdf_dF * dF_dhdotv, doesn't depend on roughness through F
        // d(d_brdf_dhdotv)/dr = d(d_brdf_dF)/dr * dF_dhdotv
        T d_d_brdf_dhdotv_dr = d_d_brdf_dF_dr * dF_dhdotv;

        T d_grad_view_i_dr = grad_out * (d_d_brdf_dndotv_dr * normal[i] +
                                          d_d_brdf_dndoth_dr * dndoth_dv[i] +
                                          d_d_brdf_dhdotv_dr * dhdotv_dv[i]);
        *grad2_roughness += gg_view[i] * d_grad_view_i_dr;

        T d_grad_light_i_dr = grad_out * (d_d_brdf_dndotl_dr * normal[i] +
                                           d_d_brdf_dndoth_dr * dndoth_dl[i] +
                                           d_d_brdf_dhdotv_dr * dhdotv_dl[i]);
        *grad2_roughness += gg_light[i] * d_grad_light_i_dr;
    }

    // =========================================================================
    // grad2_normal: gradient w.r.t. normal
    // =========================================================================

    // n_dot_l = n · l => d(n_dot_l)/dn[j] = l[j]
    // n_dot_v = n · v => d(n_dot_v)/dn[j] = v[j]
    // n_dot_h = n · h_normalized => d(n_dot_h)/dn[j] = h_normalized[j]

    // d(D)/dn[j] = dD_dndoth * h_normalized[j]
    // d(G)/dn[j] = dG_dndotv * v[j] + dG_dndotl * l[j]
    // d(denom)/dn[j] = 4 * (l[j] * n_dot_v + n_dot_l * v[j])
    // d(inv_denom)/dn[j] = -inv_denom^2 * 4 * (l[j] * n_dot_v + n_dot_l * v[j])
    //                    = -inv_denom * (l[j]/n_dot_l + v[j]/n_dot_v)

    // d(brdf)/dn[j] = d_brdf_dndotl * l[j] + d_brdf_dndotv * v[j] + d_brdf_dndoth * h_normalized[j]

    for (int j = 0; j < 3; ++j) {
        T l_j = light[j];
        T v_j = view[j];
        T h_j = h_normalized[j];

        // Derivatives of intermediate quantities w.r.t. normal[j]
        T d_ndotl_dnj = l_j;
        T d_ndotv_dnj = v_j;
        T d_ndoth_dnj = h_j;

        T d_D_dnj = dD_dndoth * h_j;
        T d_G_dnj = dG_dndotv * v_j + dG_dndotl * l_j;
        T d_inv_denom_dnj = -inv_denom * (l_j / n_dot_l_clamped + v_j / n_dot_v_clamped);

        T d_brdf_dnj = d_brdf_dndotl * l_j + d_brdf_dndotv * v_j + d_brdf_dndoth * h_j;

        // d(d_brdf_dD)/dn[j] = d(G * F / denom)/dn[j]
        //                    = dG/dn[j] * F / denom + G * F * d(inv_denom)/dn[j]
        T d_d_brdf_dD_dnj = d_G_dnj * F * inv_denom + G * F * d_inv_denom_dnj;
        // d(d_brdf_dG)/dn[j] = d(D * F / denom)/dn[j]
        T d_d_brdf_dG_dnj = d_D_dnj * F * inv_denom + D * F * d_inv_denom_dnj;
        // d(d_brdf_dF)/dn[j] = d(D * G / denom)/dn[j]
        T d_d_brdf_dF_dnj = d_D_dnj * G * inv_denom + D * d_G_dnj * inv_denom + D * G * d_inv_denom_dnj;

        // d(dG_dndotl)/dn[j] = d(G1_v * dG1_l_dndotl)/dn[j]
        //                    = dG1_v_dndotv * v_j * dG1_l_dndotl + G1_v * d(dG1_l_dndotl)/dn[j]
        // d(dG1_l_dndotl)/dn[j] = d²G1_l/d(n_dot_l)² * l_j
        T d_dG1_l_dndotl_dnj = d2G1_l_dndotl2 * l_j;
        T d_dG_dndotl_dnj = dG1_v_dndotv * v_j * dG1_l_dndotl + G1_v * d_dG1_l_dndotl_dnj;

        T d_dG1_v_dndotv_dnj = d2G1_v_dndotv2 * v_j;
        T d_dG_dndotv_dnj = d_dG1_v_dndotv_dnj * G1_l + dG1_v_dndotv * dG1_l_dndotl * l_j;

        // d(dD_dndoth)/dn[j] = d²D/d(n_dot_h)² * h_j
        T d_dD_dndoth_dnj = d2D_dndoth2 * h_j;

        // d(d_brdf_dndotl)/dn[j] = d(d_brdf_dG * dG_dndotl - brdf / n_dot_l)/dn[j]
        //                        = d(d_brdf_dG)/dn[j] * dG_dndotl + d_brdf_dG * d(dG_dndotl)/dn[j]
        //                        - d(brdf)/dn[j] / n_dot_l + brdf / n_dot_l² * l_j
        T d_d_brdf_dndotl_dnj = d_d_brdf_dG_dnj * dG_dndotl + d_brdf_dG * d_dG_dndotl_dnj
                                - d_brdf_dnj / n_dot_l_clamped + brdf / (n_dot_l_clamped * n_dot_l_clamped) * l_j;

        T d_d_brdf_dndotv_dnj = d_d_brdf_dG_dnj * dG_dndotv + d_brdf_dG * d_dG_dndotv_dnj
                                - d_brdf_dnj / n_dot_v_clamped + brdf / (n_dot_v_clamped * n_dot_v_clamped) * v_j;

        T d_d_brdf_dndoth_dnj = d_d_brdf_dD_dnj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dnj;

        // Contributions from gg_f0:
        // grad_f0 = grad_out * d_brdf_dF * dF_df0
        // d(grad_f0)/dn[j] = grad_out * d(d_brdf_dF)/dn[j] * dF_df0
        grad2_normal[j] += gg_f0 * grad_out * d_d_brdf_dF_dnj * dF_df0;

        // Contributions from gg_roughness:
        // grad_roughness depends on D, G, which depend on normal
        // d(grad_roughness)/dn[j] = grad_out * (d(d_brdf_dD)/dn[j] * dD_dalpha_sq * dalpha_sq_dr +
        //                                       d(d_brdf_dG)/dn[j] * dG_dr)
        // Note: dD_dalpha_sq depends on n_dot_h, so we need its derivative too
        // d(dD_dalpha_sq)/dn[j] = d²D/(d(alpha_sq) d(n_dot_h)) * h_j
        T d_dD_dalpha_sq_dnj = d2D_dndoth_dalpha_sq * h_j;
        T d_grad_roughness_dnj = grad_out * (d_d_brdf_dD_dnj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dnj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dnj * dG_droughness);
        grad2_normal[j] += gg_roughness * d_grad_roughness_dnj;

        // Contributions from gg_normal:
        // grad_normal[i] = grad_out * (d_brdf_dndotl * l[i] + d_brdf_dndotv * v[i] + d_brdf_dndoth * h[i])
        // d(grad_normal[i])/dn[j] = grad_out * (d(d_brdf_dndotl)/dn[j] * l[i] +
        //                                       d(d_brdf_dndotv)/dn[j] * v[i] +
        //                                       d(d_brdf_dndoth)/dn[j] * h[i])
        for (int i = 0; i < 3; ++i) {
            T d_grad_normal_i_dnj = grad_out * (d_d_brdf_dndotl_dnj * light[i] +
                                                 d_d_brdf_dndotv_dnj * view[i] +
                                                 d_d_brdf_dndoth_dnj * h_normalized[i]);
            grad2_normal[j] += gg_normal[i] * d_grad_normal_i_dnj;
        }

        // d(d_brdf_dhdotv)/dn[j] = d(d_brdf_dF * dF_dhdotv)/dn[j] = d(d_brdf_dF)/dn[j] * dF_dhdotv
        // (dF_dhdotv doesn't depend on normal)
        T d_d_brdf_dhdotv_dnj = d_d_brdf_dF_dnj * dF_dhdotv;

        // Contributions from gg_view:
        // grad_view[i] = grad_out * (d_brdf_dndotv * n[i] + d_brdf_dndoth * dndoth_dv[i] + d_brdf_dhdotv * dhdotv_dv[i])
        // d(grad_view[i])/dn[j]:
        // - d(d_brdf_dndotv * n[i])/dn[j] = d(d_brdf_dndotv)/dn[j] * n[i] + d_brdf_dndotv * delta_ij
        // - d(d_brdf_dndoth * dndoth_dv[i])/dn[j]
        //   dndoth_dv[i] = (n[i] - n_dot_h * h[i]) * inv_h_len
        //   d(dndoth_dv[i])/dn[j] = (delta_ij - d(n_dot_h)/dn[j] * h[i]) * inv_h_len
        //                         = (delta_ij - h[j] * h[i]) * inv_h_len
        T d_dndoth_dv_i_dnj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            d_dndoth_dv_i_dnj[i] = (delta_ij - h_j * h_normalized[i]) * inv_h_len;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_grad_view_i_dnj = grad_out * (d_d_brdf_dndotv_dnj * normal[i] + d_brdf_dndotv * delta_ij +
                                               d_d_brdf_dndoth_dnj * dndoth_dv[i] + d_brdf_dndoth * d_dndoth_dv_i_dnj[i] +
                                               d_d_brdf_dhdotv_dnj * dhdotv_dv[i]);
            grad2_normal[j] += gg_view[i] * d_grad_view_i_dnj;
        }

        // Contributions from gg_light:
        // grad_light[i] = grad_out * (d_brdf_dndotl * n[i] + d_brdf_dndoth * dndoth_dl[i] + d_brdf_dhdotv * dhdotv_dl[i])
        // Similar to view but with different terms
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_dndoth_dl_i_dnj = d_dndoth_dv_i_dnj[i];  // Same formula
            T d_grad_light_i_dnj = grad_out * (d_d_brdf_dndotl_dnj * normal[i] + d_brdf_dndotl * delta_ij +
                                                d_d_brdf_dndoth_dnj * dndoth_dl[i] + d_brdf_dndoth * d_dndoth_dl_i_dnj +
                                                d_d_brdf_dhdotv_dnj * dhdotv_dl[i]);
            grad2_normal[j] += gg_light[i] * d_grad_light_i_dnj;
        }
    }

    // =========================================================================
    // grad2_view: gradient w.r.t. view
    // =========================================================================

    // d(n_dot_v)/dv[j] = n[j]
    // d(h)/dv[j] = [0, 0, 0] except at component j where it's 1
    // d(h_normalized)/dv[j] = (e_j - h_normalized[j] * h_normalized) / |h|
    // d(n_dot_h)/dv[j] = n · d(h_normalized)/dv[j] = (n[j] - n_dot_h * h_normalized[j]) / |h| = dndoth_dv[j]
    // d(h_dot_v)/dv[j] = dhdotv_dv[j]

    for (int j = 0; j < 3; ++j) {
        T n_j = normal[j];
        T v_j = view[j];
        T h_j = h_normalized[j];

        // Derivatives w.r.t. view[j]
        T d_ndotv_dvj = n_j;
        T d_ndoth_dvj = dndoth_dv[j];
        T d_hdotv_dvj = dhdotv_dv[j];

        T d_D_dvj = dD_dndoth * d_ndoth_dvj;
        T d_G_dvj = dG_dndotv * n_j;
        T d_F_dvj = dF_dhdotv * d_hdotv_dvj;
        T d_inv_denom_dvj = -inv_denom * n_j / n_dot_v_clamped;

        T d_brdf_dvj = d_brdf_dndotv * n_j + d_brdf_dndoth * d_ndoth_dvj + d_brdf_dhdotv * d_hdotv_dvj;

        // d(d_brdf_dD)/dv[j] = d(G * F / denom)/dv[j]
        T d_d_brdf_dD_dvj = d_G_dvj * F * inv_denom + G * d_F_dvj * inv_denom + G * F * d_inv_denom_dvj;
        T d_d_brdf_dG_dvj = d_D_dvj * F * inv_denom + D * d_F_dvj * inv_denom + D * F * d_inv_denom_dvj;
        T d_d_brdf_dF_dvj = d_D_dvj * G * inv_denom + D * d_G_dvj * inv_denom + D * G * d_inv_denom_dvj;

        // d(dG_dndotv)/dv[j] = d(dG1_v_dndotv * G1_l)/dv[j] = d²G1_v/d(n_dot_v)² * n_j * G1_l
        T d_dG_dndotv_dvj = d2G1_v_dndotv2 * n_j * G1_l;

        // d(dD_dndoth)/dv[j] = d²D/d(n_dot_h)² * d_ndoth_dvj
        T d_dD_dndoth_dvj = d2D_dndoth2 * d_ndoth_dvj;

        // d(dF_dhdotv)/dv[j] = d²F/d(h_dot_v)² * d_hdotv_dvj
        T d_dF_dhdotv_dvj = d2F_dhdotv2 * d_hdotv_dvj;

        // d(d_brdf_dndotv)/dv[j]
        T d_d_brdf_dndotv_dvj = d_d_brdf_dG_dvj * dG_dndotv + d_brdf_dG * d_dG_dndotv_dvj
                                - d_brdf_dvj / n_dot_v_clamped + brdf / (n_dot_v_clamped * n_dot_v_clamped) * n_j;

        // d(d_brdf_dndotl)/dv[j] (dG_dndotl doesn't depend on v directly)
        T d_d_brdf_dndotl_dvj = d_d_brdf_dG_dvj * dG_dndotl - d_brdf_dvj / n_dot_l_clamped;

        T d_d_brdf_dndoth_dvj = d_d_brdf_dD_dvj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dvj;

        T d_d_brdf_dhdotv_dvj = d_d_brdf_dF_dvj * dF_dhdotv + d_brdf_dF * d_dF_dhdotv_dvj;

        // Contributions from gg_f0:
        grad2_view[j] += gg_f0 * grad_out * d_d_brdf_dF_dvj * dF_df0;

        // Contributions from gg_roughness:
        T d_dD_dalpha_sq_dvj = d2D_dndoth_dalpha_sq * d_ndoth_dvj;
        T d_grad_roughness_dvj = grad_out * (d_d_brdf_dD_dvj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dvj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dvj * dG_droughness);
        grad2_view[j] += gg_roughness * d_grad_roughness_dvj;

        // Contributions from gg_normal:
        for (int i = 0; i < 3; ++i) {
            T d_grad_normal_i_dvj = grad_out * (d_d_brdf_dndotl_dvj * light[i] +
                                                 d_d_brdf_dndotv_dvj * view[i] + d_brdf_dndotv * ((i == j) ? T(1) : T(0)) +
                                                 d_d_brdf_dndoth_dvj * h_normalized[i]);
            // Also need d(h_normalized)/dv[j]
            T d_h_normalized_i_dvj = ((i == j) ? T(1) : T(0)) * inv_h_len - h_normalized[i] * h_j * inv_h_len;
            d_grad_normal_i_dvj += grad_out * d_brdf_dndoth * d_h_normalized_i_dvj;
            grad2_view[j] += gg_normal[i] * d_grad_normal_i_dvj;
        }

        // Contributions from gg_view:
        // Need second derivatives of dndoth_dv and dhdotv_dv w.r.t. view
        // dndoth_dv[i] = (n[i] - n_dot_h * h_normalized[i]) / |h|
        // d(dndoth_dv[i])/dv[j]:
        //   d(n_dot_h)/dv[j] = dndoth_dv[j]
        //   d(h_normalized[i])/dv[j] = (delta_ij - h_i * h_j) / |h|
        //   d(1/|h|)/dv[j] = -1/|h|² * d|h|/dv[j] = -1/|h|² * h_j/|h| = -h_j / |h|³ = -h_j * inv_h_len_cu * h_len²
        //                  = -h_j * inv_h_len_sq * inv_h_len
        T d_inv_h_len_dvj = -h_j * inv_h_len_sq;
        T d2ndoth_dv_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            // d(dndoth_dv[i])/dv[j] = d[(n[i] - n_dot_h * h[i]) / |h|]/dv[j]
            // = d(n[i] - n_dot_h * h[i])/dv[j] / |h| + (n[i] - n_dot_h * h[i]) * d(1/|h|)/dv[j]
            // d(n_dot_h * h[i])/dv[j] = dndoth_dv[j] * h[i] + n_dot_h * (delta_ij - h_i * h_j) / |h|
            T d_ndoth_times_hi_dvj = dndoth_dv[j] * h_normalized[i] + n_dot_h * (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            d2ndoth_dv_i_dvj[i] = (-d_ndoth_times_hi_dvj) * inv_h_len + (normal[i] - n_dot_h * h_normalized[i]) * d_inv_h_len_dvj;
        }

        // dhdotv_dv[i] = h_normalized[i] + (v[i] - h_dot_v * h_normalized[i]) / |h|
        // d(dhdotv_dv[i])/dv[j]:
        T d2hdotv_dv_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            // d(h_normalized[i])/dv[j] = (delta_ij - h_i * h_j) / |h|
            T d_h_i_dvj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            // d[(v[i] - h_dot_v * h[i]) / |h|]/dv[j]
            // = d(v[i] - h_dot_v * h[i])/dv[j] / |h| + (v[i] - h_dot_v * h[i]) * d(1/|h|)/dv[j]
            // d(v[i])/dv[j] = delta_ij
            // d(h_dot_v * h[i])/dv[j] = dhdotv_dv[j] * h[i] + h_dot_v * d_h_i_dvj
            T d_hdotv_times_hi_dvj = dhdotv_dv[j] * h_normalized[i] + h_dot_v * d_h_i_dvj;
            T d_vi_minus_term_dvj = delta_ij - d_hdotv_times_hi_dvj;
            d2hdotv_dv_i_dvj[i] = d_h_i_dvj + d_vi_minus_term_dvj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dvj;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_grad_view_i_dvj = grad_out * (d_d_brdf_dndotv_dvj * normal[i] +
                                               d_d_brdf_dndoth_dvj * dndoth_dv[i] + d_brdf_dndoth * d2ndoth_dv_i_dvj[i] +
                                               d_d_brdf_dhdotv_dvj * dhdotv_dv[i] + d_brdf_dhdotv * d2hdotv_dv_i_dvj[i]);
            grad2_view[j] += gg_view[i] * d_grad_view_i_dvj;
        }

        // Contributions from gg_light:
        // dndoth_dl[i] = dndoth_dv[i] (same formula), d(dndoth_dl[i])/dv[j] needs separate computation
        // Actually dndoth_dl[i] = (n[i] - n_dot_h * h[i]) / |h| same as dndoth_dv[i]
        // d(dndoth_dl[i])/dv[j] = d2ndoth_dv_i_dvj[i] (same)
        // dhdotv_dl[i] = (v[i] - h_dot_v * h[i]) / |h|
        // d(dhdotv_dl[i])/dv[j]:
        T d2hdotv_dl_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dvj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            // d[(v[i] - h_dot_v * h[i]) / |h|]/dv[j]
            T d_hdotv_times_hi_dvj = dhdotv_dv[j] * h_normalized[i] + h_dot_v * d_h_i_dvj;
            T d_vi_minus_term_dvj = delta_ij - d_hdotv_times_hi_dvj;
            d2hdotv_dl_i_dvj[i] = d_vi_minus_term_dvj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dvj;
        }

        for (int i = 0; i < 3; ++i) {
            T d_grad_light_i_dvj = grad_out * (d_d_brdf_dndotl_dvj * normal[i] +
                                                d_d_brdf_dndoth_dvj * dndoth_dl[i] + d_brdf_dndoth * d2ndoth_dv_i_dvj[i] +
                                                d_d_brdf_dhdotv_dvj * dhdotv_dl[i] + d_brdf_dhdotv * d2hdotv_dl_i_dvj[i]);
            grad2_view[j] += gg_light[i] * d_grad_light_i_dvj;
        }
    }

    // =========================================================================
    // grad2_light: gradient w.r.t. light
    // =========================================================================

    // Very similar to grad2_view but with different dependency paths
    // d(n_dot_l)/dl[j] = n[j]
    // d(h_normalized)/dl[j] = (e_j - h_normalized[j] * h_normalized) / |h|
    // d(n_dot_h)/dl[j] = dndoth_dl[j]
    // d(h_dot_v)/dl[j] = dhdotv_dl[j]

    for (int j = 0; j < 3; ++j) {
        T n_j = normal[j];
        T l_j = light[j];
        T h_j = h_normalized[j];

        T d_ndotl_dlj = n_j;
        T d_ndoth_dlj = dndoth_dl[j];
        T d_hdotv_dlj = dhdotv_dl[j];

        T d_D_dlj = dD_dndoth * d_ndoth_dlj;
        T d_G_dlj = dG_dndotl * n_j;
        T d_F_dlj = dF_dhdotv * d_hdotv_dlj;
        T d_inv_denom_dlj = -inv_denom * n_j / n_dot_l_clamped;

        T d_brdf_dlj = d_brdf_dndotl * n_j + d_brdf_dndoth * d_ndoth_dlj + d_brdf_dhdotv * d_hdotv_dlj;

        T d_d_brdf_dD_dlj = d_G_dlj * F * inv_denom + G * d_F_dlj * inv_denom + G * F * d_inv_denom_dlj;
        T d_d_brdf_dG_dlj = d_D_dlj * F * inv_denom + D * d_F_dlj * inv_denom + D * F * d_inv_denom_dlj;
        T d_d_brdf_dF_dlj = d_D_dlj * G * inv_denom + D * d_G_dlj * inv_denom + D * G * d_inv_denom_dlj;

        T d_dG_dndotl_dlj = G1_v * d2G1_l_dndotl2 * n_j;
        T d_dD_dndoth_dlj = d2D_dndoth2 * d_ndoth_dlj;
        T d_dF_dhdotv_dlj = d2F_dhdotv2 * d_hdotv_dlj;

        T d_d_brdf_dndotl_dlj = d_d_brdf_dG_dlj * dG_dndotl + d_brdf_dG * d_dG_dndotl_dlj
                                - d_brdf_dlj / n_dot_l_clamped + brdf / (n_dot_l_clamped * n_dot_l_clamped) * n_j;

        T d_d_brdf_dndotv_dlj = d_d_brdf_dG_dlj * dG_dndotv - d_brdf_dlj / n_dot_v_clamped;

        T d_d_brdf_dndoth_dlj = d_d_brdf_dD_dlj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dlj;

        T d_d_brdf_dhdotv_dlj = d_d_brdf_dF_dlj * dF_dhdotv + d_brdf_dF * d_dF_dhdotv_dlj;

        // Contributions from gg_f0:
        grad2_light[j] += gg_f0 * grad_out * d_d_brdf_dF_dlj * dF_df0;

        // Contributions from gg_roughness:
        T d_dD_dalpha_sq_dlj = d2D_dndoth_dalpha_sq * d_ndoth_dlj;
        T d_grad_roughness_dlj = grad_out * (d_d_brdf_dD_dlj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dlj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dlj * dG_droughness);
        grad2_light[j] += gg_roughness * d_grad_roughness_dlj;

        // Contributions from gg_normal:
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_normalized_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_grad_normal_i_dlj = grad_out * (d_d_brdf_dndotl_dlj * light[i] + d_brdf_dndotl * delta_ij +
                                                 d_d_brdf_dndotv_dlj * view[i] +
                                                 d_d_brdf_dndoth_dlj * h_normalized[i] + d_brdf_dndoth * d_h_normalized_i_dlj);
            grad2_light[j] += gg_normal[i] * d_grad_normal_i_dlj;
        }

        // Contributions from gg_view:
        T d_inv_h_len_dlj = -h_j * inv_h_len_sq;
        T d2ndoth_dv_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_ndoth_times_hi_dlj = dndoth_dl[j] * h_normalized[i] + n_dot_h * (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            d2ndoth_dv_i_dlj[i] = (-d_ndoth_times_hi_dlj) * inv_h_len + (normal[i] - n_dot_h * h_normalized[i]) * d_inv_h_len_dlj;
        }

        T d2hdotv_dv_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dlj = dhdotv_dl[j] * h_normalized[i] + h_dot_v * d_h_i_dlj;
            T d_vi_minus_term_dlj = -d_hdotv_times_hi_dlj;
            d2hdotv_dv_i_dlj[i] = d_h_i_dlj + d_vi_minus_term_dlj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dlj;
        }

        for (int i = 0; i < 3; ++i) {
            T d_grad_view_i_dlj = grad_out * (d_d_brdf_dndotv_dlj * normal[i] +
                                               d_d_brdf_dndoth_dlj * dndoth_dv[i] + d_brdf_dndoth * d2ndoth_dv_i_dlj[i] +
                                               d_d_brdf_dhdotv_dlj * dhdotv_dv[i] + d_brdf_dhdotv * d2hdotv_dv_i_dlj[i]);
            grad2_light[j] += gg_view[i] * d_grad_view_i_dlj;
        }

        // Contributions from gg_light:
        T d2ndoth_dl_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            d2ndoth_dl_i_dlj[i] = d2ndoth_dv_i_dlj[i];  // Same formula
        }

        T d2hdotv_dl_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dlj = dhdotv_dl[j] * h_normalized[i] + h_dot_v * d_h_i_dlj;
            T d_vi_minus_term_dlj = -d_hdotv_times_hi_dlj;
            d2hdotv_dl_i_dlj[i] = d_vi_minus_term_dlj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dlj;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_grad_light_i_dlj = grad_out * (d_d_brdf_dndotl_dlj * normal[i] + d_brdf_dndotl * delta_ij +
                                                d_d_brdf_dndoth_dlj * dndoth_dl[i] + d_brdf_dndoth * d2ndoth_dl_i_dlj[i] +
                                                d_d_brdf_dhdotv_dlj * dhdotv_dl[i] + d_brdf_dhdotv * d2hdotv_dl_i_dlj[i]);
            grad2_light[j] += gg_light[i] * d_grad_light_i_dlj;
        }
    }
}

}  // namespace torchscience::impl::graphics::shading
