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
 *    c) This yields 2n poles for an order-n bandpass filter, organized as
 *       n conjugate pairs for n second-order sections.
 *
 * 2. POLE PAIRING:
 *    For each lowpass pole p_k, the transformation produces two bandpass poles:
 *      bp1_k = (B/2)*p_k + sqrt((B*p_k/2)^2 - omega_0^2)
 *      bp2_k = (B/2)*p_k - sqrt((B*p_k/2)^2 - omega_0^2)
 *
 *    For a conjugate pair of lowpass poles (p_k, p_{n-1-k} = conj(p_k)):
 *      bp1_{n-1-k} = conj(bp1_k)
 *      bp2_{n-1-k} = conj(bp2_k)
 *
 *    Therefore, (bp1_k, bp1_{n-1-k}) and (bp2_k, bp2_{n-1-k}) are conjugate pairs
 *    suitable for real SOS sections.
 *
 * 3. SOS FORMAT:
 *    Each second-order section represents:
 *    H_k(s) = (b0*s^2 + b1*s + b2) / (a0*s^2 + a1*s + a2)
 *    Output tensor: shape (..., n_sections, 6) where each row is [b0,b1,b2,a0,a1,a2]
 *
 *    For a conjugate pole pair (p, conj(p)):
 *      a0 = 1
 *      a1 = -2 * Re(p)
 *      a2 = |p|^2
 *
 * 4. DTYPE SUPPORT:
 *    - Supports float16, bfloat16, float32, float64
 *    - Complex types not directly supported (filter coefficients are real)
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <vector>
#include <array>

namespace torchscience::impl::filter {

constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// Helper structures
// ============================================================================

/**
 * Structure to hold bandpass pole information for gradient computation.
 */
template <typename T>
struct BandpassPole {
    c10::complex<T> pole;      // The bandpass pole
    c10::complex<T> lp_pole;   // The lowpass pole it came from
    c10::complex<T> sqrt_disc; // sqrt((B*p/2)^2 - omega_0^2)
    bool is_bp1;               // true if pole = half_Bp + sqrt_disc, false if pole = half_Bp - sqrt_disc
};

// ============================================================================
// Forward implementation
// ============================================================================

/**
 * Butterworth analog bandpass filter - compute SOS coefficients.
 *
 * This function computes the second-order sections for an analog Butterworth
 * bandpass filter given the filter order and passband edge frequencies.
 *
 * ALGORITHM:
 * 1. Generate n lowpass prototype poles on the unit circle
 * 2. For each conjugate pair of lowpass poles, transform to get 4 bandpass poles
 * 3. These 4 poles form 2 conjugate pairs -> 2 SOS sections
 * 4. Normalize for unity gain at center frequency
 *
 * @param n Filter order (positive integer)
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @param sos Output array of SOS coefficients, shape (n, 6)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter(
    int64_t n,
    T omega_p1,
    T omega_p2,
    T* sos  // Output: n * 6 elements
) {
    const T pi = T(kPi);

    // Compute center frequency and bandwidth
    T omega_0 = std::sqrt(omega_p1 * omega_p2);
    T omega_0_sq = omega_p1 * omega_p2;
    T B = omega_p2 - omega_p1;

    // Clamp n to max order
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Process lowpass poles to generate SOS sections
    // For each pair of conjugate lowpass poles, we get 2 SOS sections
    int64_t section_idx = 0;

    // Number of lowpass pole pairs to process
    // For even n: n/2 pairs, each giving 2 sections -> n sections total
    // For odd n: (n-1)/2 pairs + 1 middle pole -> (n-1) + 1 = n sections
    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);

    // Process conjugate pairs of lowpass poles
    for (int64_t k = 0; k < num_pairs; ++k) {
        // Lowpass pole index (1-based in the original formula)
        int64_t k1 = k + 1;

        // Compute lowpass pole angle
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        // Compute bandpass poles
        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // Section from bp1 and conj(bp1)
        // a1 = -2 * Re(bp1), a2 = |bp1|^2
        T a1_bp1 = -T(2) * bp1.real();
        T a2_bp1 = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();

        T* sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0 (will be normalized later)
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1_bp1;
        sos_row[5] = a2_bp1;
        section_idx++;

        // Section from bp2 and conj(bp2)
        T a1_bp2 = -T(2) * bp2.real();
        T a2_bp2 = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();

        sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0 (will be normalized later)
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1_bp2;
        sos_row[5] = a2_bp2;
        section_idx++;
    }

    // Handle middle pole for odd n
    if (has_middle) {
        // Middle lowpass pole is at angle = pi (real, = -1)
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        // For real lowpass pole, the two bandpass poles are either
        // both real or complex conjugates
        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // These should be conjugates of each other (or both real)
        // Either way, they form one SOS section
        // a1 = -(bp1 + bp2) = -2*half_Bp (which is real since lp_pole is real)
        // a2 = bp1 * bp2 = half_Bp^2 - disc = omega_0^2
        T a1 = -(bp1 + bp2).real();
        T a2 = (bp1 * bp2).real();

        T* sos_row = sos + section_idx * 6;
        sos_row[0] = T(1);  // b0
        sos_row[1] = T(0);  // b1
        sos_row[2] = T(0);  // b2
        sos_row[3] = T(1);  // a0
        sos_row[4] = a1;
        sos_row[5] = a2;
        section_idx++;
    }

    // Compute total gain at center frequency and normalize
    // At s = j*omega_0, for section with numerator s^2 and denominator s^2 + a1*s + a2:
    // H(j*omega_0) = (j*omega_0)^2 / ((j*omega_0)^2 + a1*(j*omega_0) + a2)
    //              = -omega_0^2 / (-omega_0^2 + j*a1*omega_0 + a2)
    //              = -omega_0^2 / ((a2 - omega_0^2) + j*a1*omega_0)
    T total_gain_sq = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        T a1 = sos_row[4];
        T a2 = sos_row[5];

        // |H(j*omega_0)|^2 = omega_0^4 / ((a2 - omega_0^2)^2 + a1^2*omega_0^2)
        T num_sq = omega_0_sq * omega_0_sq;
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        total_gain_sq *= num_sq / den_sq;
    }
    T total_gain = std::sqrt(total_gain_sq);

    // Distribute normalization factor across all sections
    T normalization_factor = T(1) / std::pow(total_gain, T(1) / T(actual_n));
    for (int64_t i = 0; i < actual_n; ++i) {
        T* sos_row = sos + i * 6;
        sos_row[0] *= normalization_factor;  // b0
    }
}

}  // namespace torchscience::impl::filter
