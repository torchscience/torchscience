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
#include <tuple>
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

// ============================================================================
// Backward implementation (first-order derivatives) - Analytical Gradients
// ============================================================================

/**
 * Backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes analytical gradients w.r.t. omega_p1 and omega_p2 given the
 * gradient w.r.t. the SOS output.
 *
 * MATHEMATICAL DERIVATION:
 * ========================
 * For a bandpass pole bp = (B/2)*p + sqrt((B*p/2)^2 - omega_0^2)
 * where p = exp(j*theta) is the lowpass pole and theta is constant:
 *
 * Let half_Bp = (B/2)*p, D = half_Bp^2 - omega_0^2, sqrt_D = sqrt(D)
 *
 * d(half_Bp)/d(omega_p1) = (-1/2)*p
 * d(half_Bp)/d(omega_p2) = (1/2)*p
 *
 * d(D)/d(omega_p1) = 2*half_Bp*(-p/2) - omega_p2 = -half_Bp*p - omega_p2
 * d(D)/d(omega_p2) = 2*half_Bp*(p/2) - omega_p1 = half_Bp*p - omega_p1
 *
 * d(sqrt_D)/d(omega) = (1/(2*sqrt_D)) * d(D)/d(omega)
 *
 * For bp1 = half_Bp + sqrt_D:
 * d(bp1)/d(omega_p1) = -p/2 + d(sqrt_D)/d(omega_p1)
 * d(bp1)/d(omega_p2) = p/2 + d(sqrt_D)/d(omega_p2)
 *
 * For bp2 = half_Bp - sqrt_D:
 * d(bp2)/d(omega_p1) = -p/2 - d(sqrt_D)/d(omega_p1)
 * d(bp2)/d(omega_p2) = p/2 - d(sqrt_D)/d(omega_p2)
 *
 * For SOS coefficients:
 * a1 = -2 * Re(bp)
 * a2 = |bp|^2 = Re(bp)^2 + Im(bp)^2
 *
 * d(a1)/d(omega) = -2 * Re(d(bp)/d(omega))
 * d(a2)/d(omega) = 2 * Re(bp * conj(d(bp)/d(omega)))
 *
 * For the normalization factor b0 = nf, we use the chain rule through
 * the gain computation.
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
    const T pi = T(kPi);
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Compute intermediate quantities
    T omega_0_sq = omega_p1 * omega_p2;
    T omega_0 = std::sqrt(omega_0_sq);
    T B = omega_p2 - omega_p1;

    // We need to recompute the forward pass to get the poles and normalization
    // This is necessary for correct gradient computation

    // First, compute all poles and their derivatives
    struct PoleInfo {
        c10::complex<T> bp;       // bandpass pole
        c10::complex<T> lp;       // lowpass pole
        c10::complex<T> sqrt_disc;// sqrt of discriminant
        bool is_bp1;              // true for bp1, false for bp2
    };

    PoleInfo poles[kMaxOrder];
    T a1_vals[kMaxOrder];
    T a2_vals[kMaxOrder];

    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);
    int64_t section_idx = 0;

    // Compute poles for each section
    for (int64_t k = 0; k < num_pairs; ++k) {
        int64_t k1 = k + 1;
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // Section from bp1
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, true};
        a1_vals[section_idx] = -T(2) * bp1.real();
        a2_vals[section_idx] = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();
        section_idx++;

        // Section from bp2
        poles[section_idx] = {bp2, lp_pole, sqrt_disc, false};
        a1_vals[section_idx] = -T(2) * bp2.real();
        a2_vals[section_idx] = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();
        section_idx++;
    }

    if (has_middle) {
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // For middle pole, we use bp1 and bp2 together in one section
        // a1 = -(bp1 + bp2) = -2*half_Bp
        // a2 = bp1 * bp2 = omega_0^2
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, true};  // Store bp1, derivative is special
        a1_vals[section_idx] = -(bp1 + bp2).real();
        a2_vals[section_idx] = (bp1 * bp2).real();
        section_idx++;
    }

    // Compute gain and normalization factor
    T total_gain_sq = T(1);
    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T num_sq = omega_0_sq * omega_0_sq;
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        total_gain_sq *= num_sq / den_sq;
    }
    T total_gain = std::sqrt(total_gain_sq);
    T nf = T(1) / std::pow(total_gain, T(1) / T(actual_n));

    // Now compute gradients
    grad_omega_p1 = T(0);
    grad_omega_p2 = T(0);

    // Gradient from a1 and a2 coefficients (before normalization effects)
    for (int64_t i = 0; i < actual_n; ++i) {
        const T* grad_row = grad_sos + i * 6;
        T grad_b0 = grad_row[0];
        T grad_a1 = grad_row[4];
        T grad_a2 = grad_row[5];

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;

        // Handle middle pole specially
        bool is_middle = has_middle && (i == actual_n - 1);

        if (is_middle) {
            // For middle pole: a1 = -B*lp.real(), a2 = omega_0^2
            // Since lp is real (cos(pi) = -1), lp.real() = lp
            T lp_real = lp.real();

            // d(a1)/d(omega_p1) = -d(B)/d(omega_p1) * lp_real = lp_real
            // d(a1)/d(omega_p2) = -d(B)/d(omega_p2) * lp_real = -lp_real
            grad_omega_p1 += grad_a1 * lp_real;
            grad_omega_p2 += grad_a1 * (-lp_real);

            // d(a2)/d(omega_p1) = omega_p2
            // d(a2)/d(omega_p2) = omega_p1
            grad_omega_p1 += grad_a2 * omega_p2;
            grad_omega_p2 += grad_a2 * omega_p1;
        } else {
            // For conjugate-paired poles:
            // bp = (B/2)*lp + sign * sqrt((B*lp/2)^2 - omega_0^2)
            // where sign = +1 for bp1, -1 for bp2

            c10::complex<T> half_Bp = (B * lp) / T(2);

            // d(half_Bp)/d(omega_p1) = -lp/2
            // d(half_Bp)/d(omega_p2) = lp/2
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);

            // d(D)/d(omega_p1) = -half_Bp*lp - omega_p2
            // d(D)/d(omega_p2) = half_Bp*lp - omega_p1
            c10::complex<T> d_D_d_w1 = -half_Bp * lp - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = half_Bp * lp - c10::complex<T>(omega_p1, T(0));

            // d(sqrt_D)/d(omega) = d(D)/d(omega) / (2 * sqrt_D)
            c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
            c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
            c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

            // d(bp)/d(omega) depends on whether it's bp1 or bp2
            c10::complex<T> d_bp_d_w1, d_bp_d_w2;
            if (is_bp1) {
                d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
            } else {
                d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
            }

            // d(a1)/d(omega) = -2 * Re(d(bp)/d(omega))
            T d_a1_d_w1 = -T(2) * d_bp_d_w1.real();
            T d_a1_d_w2 = -T(2) * d_bp_d_w2.real();

            // d(a2)/d(omega) = 2 * Re(bp * conj(d(bp)/d(omega)))
            // Re(bp * conj(d_bp)) = Re(bp)*Re(d_bp) + Im(bp)*Im(d_bp)
            T d_a2_d_w1 = T(2) * (bp.real() * d_bp_d_w1.real() + bp.imag() * d_bp_d_w1.imag());
            T d_a2_d_w2 = T(2) * (bp.real() * d_bp_d_w2.real() + bp.imag() * d_bp_d_w2.imag());

            grad_omega_p1 += grad_a1 * d_a1_d_w1 + grad_a2 * d_a2_d_w1;
            grad_omega_p2 += grad_a1 * d_a1_d_w2 + grad_a2 * d_a2_d_w2;
        }

        // Gradient from b0 (normalization factor)
        // b0 = nf for all sections
        // nf = 1 / total_gain^(1/n)
        // total_gain = sqrt(prod_i |H_i(j*omega_0)|^2)
        // |H_i|^2 = omega_0^4 / ((a2_i - omega_0^2)^2 + a1_i^2 * omega_0^2)
        //
        // This requires computing d(nf)/d(omega) which involves the chain rule
        // through all the gain terms. For simplicity, we compute this separately.
    }

    // Compute gradient of normalization factor
    // nf = total_gain^(-1/n)
    // d(nf)/d(omega) = (-1/n) * total_gain^(-1/n - 1) * d(total_gain)/d(omega)
    //                = (-1/n) * nf / total_gain * d(total_gain)/d(omega)
    //                = (-nf / (n * total_gain)) * d(total_gain)/d(omega)
    //
    // total_gain = sqrt(total_gain_sq)
    // d(total_gain)/d(omega) = d(total_gain_sq)/d(omega) / (2 * total_gain)
    //
    // total_gain_sq = prod_i (omega_0^4 / den_sq_i)
    // log(total_gain_sq) = sum_i (4*log(omega_0) - log(den_sq_i))
    // d(log(total_gain_sq))/d(omega) = 4*n*d(log(omega_0))/d(omega) - sum_i d(log(den_sq_i))/d(omega)
    //                                = 2*n/omega_0 * d(omega_0)/d(omega) - sum_i (1/den_sq_i)*d(den_sq_i)/d(omega)

    // d(omega_0)/d(omega_p1) = omega_p2 / (2*omega_0)
    // d(omega_0)/d(omega_p2) = omega_p1 / (2*omega_0)
    T d_omega0_d_w1 = omega_p2 / (T(2) * omega_0);
    T d_omega0_d_w2 = omega_p1 / (T(2) * omega_0);

    // Compute d(log(total_gain_sq))/d(omega)
    // log(total_gain_sq) = sum_i (4*log(omega_0) - log(den_sq_i))
    //                    = 4*n*log(omega_0) - sum_i log(den_sq_i)
    // d(log(total_gain_sq))/d(omega) = 4*n/omega_0 * d(omega_0)/d(omega) - sum_i (1/den_sq_i) * d(den_sq_i)/d(omega)
    T d_log_gain_sq_d_w1 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w1;
    T d_log_gain_sq_d_w2 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w2;

    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;

        // d(den_sq)/d(omega) = 2*den_real*d(den_real)/d(omega) + 2*a1*omega_0_sq*d(a1)/d(omega) + a1^2*2*omega_0*d(omega_0)/d(omega)
        // d(den_real)/d(omega) = d(a2)/d(omega) - 2*omega_0*d(omega_0)/d(omega)

        // We need d(a1)/d(omega) and d(a2)/d(omega) for this section
        // These are the same as computed above, but we need to store or recompute them

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;
        bool is_middle = has_middle && (i == actual_n - 1);

        T d_a1_d_w1, d_a1_d_w2, d_a2_d_w1, d_a2_d_w2;

        if (is_middle) {
            T lp_real = lp.real();
            d_a1_d_w1 = lp_real;
            d_a1_d_w2 = -lp_real;
            d_a2_d_w1 = omega_p2;
            d_a2_d_w2 = omega_p1;
        } else {
            c10::complex<T> half_Bp = (B * lp) / T(2);
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = -half_Bp * lp - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = half_Bp * lp - c10::complex<T>(omega_p1, T(0));
            c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
            c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
            c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

            c10::complex<T> d_bp_d_w1, d_bp_d_w2;
            if (is_bp1) {
                d_bp_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;
            } else {
                d_bp_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
                d_bp_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;
            }

            d_a1_d_w1 = -T(2) * d_bp_d_w1.real();
            d_a1_d_w2 = -T(2) * d_bp_d_w2.real();
            d_a2_d_w1 = T(2) * (bp.real() * d_bp_d_w1.real() + bp.imag() * d_bp_d_w1.imag());
            d_a2_d_w2 = T(2) * (bp.real() * d_bp_d_w2.real() + bp.imag() * d_bp_d_w2.imag());
        }

        // d(den_real)/d(omega) = d(a2)/d(omega) - 2*omega_0*d(omega_0)/d(omega)
        T d_den_real_d_w1 = d_a2_d_w1 - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2 - T(2) * omega_0 * d_omega0_d_w2;

        // d(den_sq)/d(omega) = 2*den_real*d(den_real)/d(omega) + 2*a1*omega_0_sq*d(a1)/d(omega) + 2*a1^2*omega_0*d(omega_0)/d(omega)
        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        // Subtract contribution to d(log(total_gain_sq))
        d_log_gain_sq_d_w1 -= d_den_sq_d_w1 / den_sq;
        d_log_gain_sq_d_w2 -= d_den_sq_d_w2 / den_sq;
    }

    // d(total_gain_sq)/d(omega) = total_gain_sq * d(log(total_gain_sq))/d(omega)
    // d(total_gain)/d(omega) = d(total_gain_sq)/d(omega) / (2*total_gain)
    //                        = total_gain_sq * d(log(total_gain_sq))/d(omega) / (2*total_gain)
    //                        = total_gain * d(log(total_gain_sq))/d(omega) / 2
    T d_total_gain_d_w1 = total_gain * d_log_gain_sq_d_w1 / T(2);
    T d_total_gain_d_w2 = total_gain * d_log_gain_sq_d_w2 / T(2);

    // d(nf)/d(omega) = (-nf / (n * total_gain)) * d(total_gain)/d(omega)
    T d_nf_d_w1 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w1;
    T d_nf_d_w2 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w2;

    // Add contribution from grad_b0 for all sections
    T total_grad_b0 = T(0);
    for (int64_t i = 0; i < actual_n; ++i) {
        total_grad_b0 += grad_sos[i * 6];
    }

    grad_omega_p1 += total_grad_b0 * d_nf_d_w1;
    grad_omega_p2 += total_grad_b0 * d_nf_d_w2;
}

// ============================================================================
// Double-backward implementation (second-order derivatives)
// ============================================================================

/**
 * Double-backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes second-order gradients for Hessian-vector products.
 * Currently returns zeros - full implementation would require
 * computing second derivatives of all the intermediate quantities.
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
