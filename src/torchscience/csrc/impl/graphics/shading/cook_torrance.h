// src/torchscience/csrc/impl/graphics/shading/cook_torrance.h
#pragma once

/*
 * Cook-Torrance BRDF Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The Cook-Torrance specular BRDF is:
 *
 *   f_r = D * F * G / (4 * (n·l) * (n·v))
 *
 * Where:
 *   D = GGX/Trowbridge-Reitz normal distribution
 *   F = Schlick Fresnel approximation
 *   G = Schlick-GGX geometry with Smith masking-shadowing
 *
 * COMPONENT FORMULAS:
 * ===================
 *
 * GGX Distribution (D):
 *   D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)
 *   where α = roughness²
 *
 * Schlick-GGX Geometry (G):
 *   G(l, v) = G₁(l) * G₁(v)
 *   G₁(x) = (n·x) / ((n·x)(1 - k) + k)
 *   k = (roughness + 1)² / 8  (for direct lighting)
 *
 * Schlick Fresnel (F):
 *   F(h, v) = F₀ + (1 - F₀) * (1 - h·v)⁵
 */

#include <c10/macros/Macros.h>
#include <cmath>

namespace torchscience::impl::graphics::shading {

// Minimum roughness to avoid division by zero
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T min_roughness() { return T(0.001); }

// Small epsilon for dot product clamping
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot_epsilon() { return T(1e-7); }

/**
 * Compute GGX/Trowbridge-Reitz normal distribution function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T ggx_distribution(T n_dot_h, T alpha_squared) {
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    denom = denom * denom;
    const T PI = T(3.14159265358979323846);
    return alpha_squared / (PI * denom);
}

/**
 * Compute Schlick-GGX geometry sub-term.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_g1(T n_dot_x, T k) {
    return n_dot_x / (n_dot_x * (T(1) - k) + k);
}

/**
 * Compute Schlick-GGX geometry term (Smith masking-shadowing).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_geometry(T n_dot_v, T n_dot_l, T roughness) {
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    return schlick_ggx_g1(n_dot_v, k) * schlick_ggx_g1(n_dot_l, k);
}

/**
 * Compute Schlick Fresnel approximation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_fresnel(T h_dot_v, T f0) {
    T one_minus_cos = T(1) - h_dot_v;
    T pow5 = one_minus_cos * one_minus_cos;
    pow5 = pow5 * pow5 * one_minus_cos;
    return f0 + (T(1) - f0) * pow5;
}

/**
 * Compute dot product of two 3D vectors.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot3(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Compute Cook-Torrance specular BRDF for a single sample.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T cook_torrance_scalar(
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0
) {
    // Clamp roughness to avoid singularities
    roughness = std::max(roughness, min_roughness<T>());

    // Compute dot products
    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    // Early out for back-facing geometry
    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return T(0);
    }

    n_dot_l = std::max(n_dot_l, dot_epsilon<T>());
    n_dot_v = std::max(n_dot_v, dot_epsilon<T>());

    // Compute halfway vector: h = normalize(l + v)
    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < dot_epsilon<T>()) {
        return T(0);
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), T(0));
    T h_dot_v = std::max(dot3(h_normalized, view), T(0));

    // Compute BRDF components
    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    T D = ggx_distribution(n_dot_h, alpha_squared);
    T G = schlick_ggx_geometry(n_dot_v, n_dot_l, roughness);
    T F = schlick_fresnel(h_dot_v, f0);

    // Cook-Torrance specular BRDF
    T denominator = T(4) * n_dot_l * n_dot_v;
    return (D * G * F) / denominator;
}

}  // namespace torchscience::impl::graphics::shading
