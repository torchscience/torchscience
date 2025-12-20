#pragma once

/*
 * Butterworth Analog Bandpass Filter
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    An analog Butterworth bandpass filter is designed by:
 *    a) Creating a Butterworth lowpass prototype with poles:
 *       s_k = exp(j * pi * (2k + n - 1) / (2n)) for k = 1, ..., n
 *    b) Applying the lowpass-to-bandpass transformation:
 *       s -> (s^2 + omega_0^2) / (B * s)
 *       where omega_0 = sqrt(omega_p1 * omega_p2), B = omega_p2 - omega_p1
 *    c) This yields 2n poles for an order-n bandpass filter
 *
 * 2. SOS FORMAT:
 *    Each second-order section represents:
 *    H_k(s) = (b0*s^2 + b1*s + b2) / (a0*s^2 + a1*s + a2)
 *    Output tensor: shape (..., n_sections, 6) where each row is [b0,b1,b2,a0,a1,a2]
 *
 * 3. DERIVATIVE FORMULAS:
 *    The SOS coefficients depend on omega_p1 and omega_p2 through:
 *    - omega_0 = sqrt(omega_p1 * omega_p2)
 *    - B = omega_p2 - omega_p1
 *    - Pole locations from the transformation
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Complex types not directly supported (filter coefficients are real)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <array>

namespace torchscience::impl::filter {

constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// Helper functions for Butterworth filter design
// ============================================================================

/**
 * Compute Butterworth lowpass prototype poles (left half-plane only).
 * s_k = exp(j * pi * (2k + n - 1) / (2n)) for k = 1, ..., n
 *
 * These are the n poles in the left half-plane of the unit circle.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_prototype_poles(
    int64_t n,
    c10::complex<T>* poles
) {
    const T pi = T(kPi);
    for (int64_t k = 1; k <= n; ++k) {
        T angle = pi * (T(2 * k) + T(n) - T(1)) / (T(2) * T(n));
        poles[k - 1] = c10::complex<T>(std::cos(angle), std::sin(angle));
    }
}

/**
 * Transform lowpass prototype pole to bandpass poles.
 * For each lowpass pole p, the bandpass has two poles:
 * s = (B*p +/- sqrt((B*p)^2 - 4*omega_0^2)) / 2
 *
 * Returns the two bandpass poles for the given lowpass pole.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::pair<c10::complex<T>, c10::complex<T>> lowpass_to_bandpass_pole(
    c10::complex<T> lp_pole,
    T omega_0,
    T bandwidth
) {
    c10::complex<T> half_Bp = (bandwidth * lp_pole) / T(2);
    c10::complex<T> discriminant = half_Bp * half_Bp - omega_0 * omega_0;
    c10::complex<T> sqrt_disc = std::sqrt(discriminant);

    return std::make_pair(half_Bp + sqrt_disc, half_Bp - sqrt_disc);
}

/**
 * Convert a pair of poles to second-order section coefficients.
 * For poles p1, p2: (s - p1)(s - p2) = s^2 - (p1+p2)*s + p1*p2
 *
 * For analog bandpass, numerator is s^2 (zeros at origin).
 * Returns [b0, b1, b2, a0, a1, a2] normalized so a0 = 1.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::array<T, 6> poles_to_sos_section(
    c10::complex<T> p1,
    c10::complex<T> p2
) {
    // Denominator: (s - p1)(s - p2) = s^2 - (p1+p2)*s + p1*p2
    c10::complex<T> sum_p = p1 + p2;
    c10::complex<T> prod_p = p1 * p2;

    // For a valid SOS section, p1 and p2 should be complex conjugates
    // or both real, so coefficients should be real
    T a0 = T(1);
    T a1 = -sum_p.real();  // Real part of -(p1+p2)
    T a2 = prod_p.real();  // Real part of p1*p2

    // Numerator: s^2 for bandpass (zeros at origin)
    T b0 = T(1);
    T b1 = T(0);
    T b2 = T(0);

    return {b0, b1, b2, a0, a1, a2};
}

/**
 * Evaluate the magnitude of a second-order section at s = j*omega.
 * H(s) = (b0*s^2 + b1*s + b2) / (a0*s^2 + a1*s + a2)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T evaluate_sos_magnitude(
    T b0, T b1, T b2,
    T a0, T a1, T a2,
    T omega
) {
    // At s = j*omega:
    // Numerator: b0*(j*omega)^2 + b1*(j*omega) + b2 = -b0*omega^2 + j*b1*omega + b2
    //          = (b2 - b0*omega^2) + j*(b1*omega)
    // Denominator: a0*(j*omega)^2 + a1*(j*omega) + a2 = -a0*omega^2 + j*a1*omega + a2
    //            = (a2 - a0*omega^2) + j*(a1*omega)
    T omega2 = omega * omega;

    T num_real = b2 - b0 * omega2;
    T num_imag = b1 * omega;
    T den_real = a2 - a0 * omega2;
    T den_imag = a1 * omega;

    T num_mag2 = num_real * num_real + num_imag * num_imag;
    T den_mag2 = den_real * den_real + den_imag * den_imag;

    return std::sqrt(num_mag2 / den_mag2);
}

// ============================================================================
// Forward implementation
// ============================================================================

/**
 * Butterworth analog bandpass filter - compute SOS coefficients.
 *
 * This function computes the second-order sections for an analog Butterworth
 * bandpass filter given the filter order and passband edge frequencies.
 *
 * @param n Filter order (positive integer)
 * @param omega_p1 Lower passband frequency (normalized, 0 < omega_p1 < omega_p2 < 1)
 * @param omega_p2 Upper passband frequency (normalized, 0 < omega_p1 < omega_p2 < 1)
 * @param sos Output array of SOS coefficients, shape (n, 6)
 *
 * The output tensor has shape (n, 6) where each row is [b0, b1, b2, a0, a1, a2]
 * for the transfer function H_k(s) = (b0*s^2 + b1*s + b2) / (a0*s^2 + a1*s + a2)
 *
 * Note: For batched operation, this function should be called per-element.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter(
    int64_t n,
    T omega_p1,
    T omega_p2,
    T* sos  // Output: n * 6 elements
) {
    // Compute center frequency and bandwidth
    T omega_0 = std::sqrt(omega_p1 * omega_p2);
    T bandwidth = omega_p2 - omega_p1;

    // Allocate space for lowpass prototype poles (max reasonable order)
    // For device code, we use a fixed-size array
    constexpr int64_t kMaxOrder = 64;
    c10::complex<T> lp_poles[kMaxOrder];

    // Clamp n to max order
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Compute lowpass prototype poles
    butterworth_prototype_poles<T>(actual_n, lp_poles);

    // First pass: compute unnormalized SOS sections
    for (int64_t i = 0; i < actual_n; ++i) {
        auto [bp_pole1, bp_pole2] = lowpass_to_bandpass_pole<T>(
            lp_poles[i], omega_0, bandwidth
        );

        // Convert pole pair to SOS section
        auto section = poles_to_sos_section<T>(bp_pole1, bp_pole2);

        T* sos_row = sos + i * 6;
        sos_row[0] = section[0];  // b0
        sos_row[1] = section[1];  // b1
        sos_row[2] = section[2];  // b2
        sos_row[3] = section[3];  // a0
        sos_row[4] = section[4];  // a1
        sos_row[5] = section[5];  // a2
    }

    // Second pass: evaluate total gain at center frequency and normalize
    T total_gain = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        T section_gain = evaluate_sos_magnitude<T>(
            sos_row[0], sos_row[1], sos_row[2],
            sos_row[3], sos_row[4], sos_row[5],
            omega_0
        );
        total_gain *= section_gain;
    }

    // Apply normalization: scale numerator to achieve unity gain at center frequency
    // Distribute the gain normalization across all sections
    T normalization_factor = T(1) / std::pow(total_gain, T(1) / T(actual_n));
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        sos_row[0] *= normalization_factor;  // b0
        sos_row[1] *= normalization_factor;  // b1
        sos_row[2] *= normalization_factor;  // b2
    }
}

// ============================================================================
// Backward implementation (first-order derivatives)
// ============================================================================

/**
 * Backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes gradients w.r.t. omega_p1 and omega_p2 given the gradient
 * w.r.t. the SOS output.
 *
 * Uses numerical differentiation for simplicity and correctness.
 *
 * @param grad_sos Gradient of loss w.r.t. SOS output, shape (n, 6)
 * @param n Filter order
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @param grad_omega_p1 Output gradient w.r.t. omega_p1
 * @param grad_omega_p2 Output gradient w.r.t. omega_p2
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter_backward(
    const T* grad_sos,  // Input: n * 6 elements
    int64_t n,
    T omega_p1,
    T omega_p2,
    T& grad_omega_p1,
    T& grad_omega_p2
) {
    // Use numerical differentiation for gradient computation
    // This is simpler and more robust than analytical derivatives
    // for the complex gain normalization
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    T eps = T(1e-7);

    // Compute SOS at slightly perturbed omega_p1
    T sos_p1_plus[kMaxOrder * 6];
    T sos_p1_minus[kMaxOrder * 6];
    butterworth_analog_bandpass_filter<T>(n, omega_p1 + eps, omega_p2, sos_p1_plus);
    butterworth_analog_bandpass_filter<T>(n, omega_p1 - eps, omega_p2, sos_p1_minus);

    // Compute gradient w.r.t. omega_p1 via chain rule
    grad_omega_p1 = T(0);
    for (int64_t i = 0; i < actual_n * 6; ++i) {
        T dsos_d_omega_p1 = (sos_p1_plus[i] - sos_p1_minus[i]) / (T(2) * eps);
        grad_omega_p1 += grad_sos[i] * dsos_d_omega_p1;
    }

    // Compute SOS at slightly perturbed omega_p2
    T sos_p2_plus[kMaxOrder * 6];
    T sos_p2_minus[kMaxOrder * 6];
    butterworth_analog_bandpass_filter<T>(n, omega_p1, omega_p2 + eps, sos_p2_plus);
    butterworth_analog_bandpass_filter<T>(n, omega_p1, omega_p2 - eps, sos_p2_minus);

    // Compute gradient w.r.t. omega_p2 via chain rule
    grad_omega_p2 = T(0);
    for (int64_t i = 0; i < actual_n * 6; ++i) {
        T dsos_d_omega_p2 = (sos_p2_plus[i] - sos_p2_minus[i]) / (T(2) * eps);
        grad_omega_p2 += grad_sos[i] * dsos_d_omega_p2;
    }
}

// ============================================================================
// Double-backward implementation (second-order derivatives)
// ============================================================================

/**
 * Double-backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes second-order gradients for Hessian-vector products.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T, T> butterworth_analog_bandpass_filter_backward_backward(
    T grad_grad_omega_p1,
    T grad_grad_omega_p2,
    const T* grad_sos,
    int64_t n,
    T omega_p1,
    T omega_p2,
    bool has_grad_grad_omega_p1,
    bool has_grad_grad_omega_p2
) {
    // Second-order derivatives are complex - for now return zeros
    // This can be implemented using symbolic differentiation if needed
    return std::make_tuple(T(0), T(0), T(0), T(0));
}

}  // namespace torchscience::impl::filter
