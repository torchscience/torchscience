#pragma once

#include "butterworth_analog_bandpass_filter.h"
#include <cmath>
#include <limits>

namespace torchscience::impl::filter {

// ============================================================================
// Double-backward implementation (second-order derivatives)
// ============================================================================

/**
 * Double-backward pass for butterworth_analog_bandpass_filter.
 *
 * Computes second-order gradients for Hessian-vector products.
 *
 * MATHEMATICAL DERIVATION:
 * ========================
 * The first backward computes:
 *   grad_omega_p1 = sum_i(grad_sos[i] * J_1i)  where J_1i = d(sos[i])/d(omega_p1)
 *   grad_omega_p2 = sum_i(grad_sos[i] * J_2i)  where J_2i = d(sos[i])/d(omega_p2)
 *
 * The double backward computes gradients w.r.t. inputs of the first backward:
 *   grad_grad_sos[i] = grad_grad_omega_p1 * J_1i + grad_grad_omega_p2 * J_2i
 *   new_grad_omega_p1 = sum_i(grad_sos[i] * H_11i * gg_w1 + grad_sos[i] * H_21i * gg_w2)
 *   new_grad_omega_p2 = sum_i(grad_sos[i] * H_12i * gg_w1 + grad_sos[i] * H_22i * gg_w2)
 *
 * where H_jki = d²(sos[i])/d(omega_pj)d(omega_pk) is the Hessian.
 *
 * For bandpass poles:
 *   bp = half_Bp ± sqrt(D)
 *   half_Bp = (B/2) * p = ((omega_p2 - omega_p1)/2) * p
 *   D = half_Bp² - omega_0² = half_Bp² - omega_p1 * omega_p2
 *
 * Second derivatives of D:
 *   d²D/d(w1)² = p²/2
 *   d²D/d(w2)² = p²/2
 *   d²D/d(w1)d(w2) = -p²/2 - 1
 *
 * @param grad_grad_omega_p1 Gradient w.r.t. grad_omega_p1 from backward
 * @param grad_grad_omega_p2 Gradient w.r.t. grad_omega_p2 from backward
 * @param grad_sos Original gradient w.r.t. SOS output, shape (n, 6)
 * @param n Filter order
 * @param omega_p1 Lower passband frequency
 * @param omega_p2 Upper passband frequency
 * @param has_grad_grad_omega_p1 Whether grad_grad_omega_p1 is non-zero
 * @param has_grad_grad_omega_p2 Whether grad_grad_omega_p2 is non-zero
 * @param grad_grad_sos Output: gradient w.r.t. grad_sos, shape (n, 6)
 * @param new_grad_omega_p1 Output: new gradient w.r.t. omega_p1
 * @param new_grad_omega_p2 Output: new gradient w.r.t. omega_p2
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void butterworth_analog_bandpass_filter_backward_backward(
    T grad_grad_omega_p1,
    T grad_grad_omega_p2,
    const T* grad_sos,
    int64_t n,
    T omega_p1,
    T omega_p2,
    bool has_grad_grad_omega_p1,
    bool has_grad_grad_omega_p2,
    T* grad_grad_sos,  // Output: n * 6 elements
    T& new_grad_omega_p1,
    T& new_grad_omega_p2
) {
    const T pi = T(kPi);
    constexpr int64_t kMaxOrder = 64;
    int64_t actual_n = n < kMaxOrder ? n : kMaxOrder;

    // Initialize outputs
    new_grad_omega_p1 = T(0);
    new_grad_omega_p2 = T(0);
    for (int64_t i = 0; i < actual_n * 6; ++i) {
        grad_grad_sos[i] = T(0);
    }

    // Early exit if no gradients
    if (!has_grad_grad_omega_p1 && !has_grad_grad_omega_p2) {
        return;
    }

    // Compute intermediate quantities
    T omega_0_sq = omega_p1 * omega_p2;
    T omega_0 = std::sqrt(omega_0_sq);
    T B = omega_p2 - omega_p1;

    // Derivatives of omega_0 and omega_0_sq
    T d_omega0_d_w1 = omega_p2 / (T(2) * omega_0);
    T d_omega0_d_w2 = omega_p1 / (T(2) * omega_0);

    // Second derivatives of omega_0
    // omega_0 = sqrt(omega_p1 * omega_p2)
    // d²omega_0/dw1² = d/dw1[w2/(2*sqrt(w1*w2))] = -w2²/(4*(w1*w2)^(3/2))
    // d²omega_0/dw2² = d/dw2[w1/(2*sqrt(w1*w2))] = -w1²/(4*(w1*w2)^(3/2))
    // d²omega_0/dw1dw2 = d/dw2[w2/(2*sqrt(w1*w2))] = 1/(2*sqrt(w1*w2)) - w2*w1/(4*(w1*w2)^(3/2))
    //                  = 1/(2*omega_0) - omega_0/(4*omega_0_sq) = 1/(4*omega_0)
    T omega_0_cubed = omega_0 * omega_0_sq;
    T d2_omega0_d_w1_w1 = -omega_p2 * omega_p2 / (T(4) * omega_0_cubed);
    T d2_omega0_d_w2_w2 = -omega_p1 * omega_p1 / (T(4) * omega_0_cubed);
    T d2_omega0_d_w1_w2 = T(1) / (T(4) * omega_0);

    // Recompute pole information
    struct PoleInfo {
        c10::complex<T> bp;
        c10::complex<T> lp;
        c10::complex<T> sqrt_disc;
        c10::complex<T> half_Bp;
        bool is_bp1;
    };

    PoleInfo poles[kMaxOrder];
    T a1_vals[kMaxOrder];
    T a2_vals[kMaxOrder];

    // First derivatives storage
    T d_a1_d_w1[kMaxOrder], d_a1_d_w2[kMaxOrder];
    T d_a2_d_w1[kMaxOrder], d_a2_d_w2[kMaxOrder];

    int64_t num_pairs = actual_n / 2;
    bool has_middle = (actual_n % 2 == 1);
    int64_t section_idx = 0;

    // Compute poles and first derivatives for each section
    for (int64_t k = 0; k < num_pairs; ++k) {
        int64_t k1 = k + 1;
        T angle = pi * (T(2 * k1) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);

        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        // First derivatives of half_Bp
        c10::complex<T> d_half_Bp_d_w1 = -lp_pole / T(2);
        c10::complex<T> d_half_Bp_d_w2 = lp_pole / T(2);

        // First derivatives of disc D = half_Bp² - omega_0²
        c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
        c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

        // First derivatives of sqrt_disc
        c10::complex<T> two_sqrt_disc = T(2) * sqrt_disc;
        c10::complex<T> d_sqrt_d_w1 = d_D_d_w1 / two_sqrt_disc;
        c10::complex<T> d_sqrt_d_w2 = d_D_d_w2 / two_sqrt_disc;

        // Section from bp1
        poles[section_idx] = {bp1, lp_pole, sqrt_disc, half_Bp, true};
        a1_vals[section_idx] = -T(2) * bp1.real();
        a2_vals[section_idx] = bp1.real() * bp1.real() + bp1.imag() * bp1.imag();

        c10::complex<T> d_bp1_d_w1 = d_half_Bp_d_w1 + d_sqrt_d_w1;
        c10::complex<T> d_bp1_d_w2 = d_half_Bp_d_w2 + d_sqrt_d_w2;

        d_a1_d_w1[section_idx] = -T(2) * d_bp1_d_w1.real();
        d_a1_d_w2[section_idx] = -T(2) * d_bp1_d_w2.real();
        d_a2_d_w1[section_idx] = T(2) * (bp1.real() * d_bp1_d_w1.real() + bp1.imag() * d_bp1_d_w1.imag());
        d_a2_d_w2[section_idx] = T(2) * (bp1.real() * d_bp1_d_w2.real() + bp1.imag() * d_bp1_d_w2.imag());
        section_idx++;

        // Section from bp2
        poles[section_idx] = {bp2, lp_pole, sqrt_disc, half_Bp, false};
        a1_vals[section_idx] = -T(2) * bp2.real();
        a2_vals[section_idx] = bp2.real() * bp2.real() + bp2.imag() * bp2.imag();

        c10::complex<T> d_bp2_d_w1 = d_half_Bp_d_w1 - d_sqrt_d_w1;
        c10::complex<T> d_bp2_d_w2 = d_half_Bp_d_w2 - d_sqrt_d_w2;

        d_a1_d_w1[section_idx] = -T(2) * d_bp2_d_w1.real();
        d_a1_d_w2[section_idx] = -T(2) * d_bp2_d_w2.real();
        d_a2_d_w1[section_idx] = T(2) * (bp2.real() * d_bp2_d_w1.real() + bp2.imag() * d_bp2_d_w1.imag());
        d_a2_d_w2[section_idx] = T(2) * (bp2.real() * d_bp2_d_w2.real() + bp2.imag() * d_bp2_d_w2.imag());
        section_idx++;
    }

    if (has_middle) {
        int64_t k_mid = num_pairs + 1;
        T angle = pi * (T(2 * k_mid) + T(actual_n) - T(1)) / (T(2) * T(actual_n));
        c10::complex<T> lp_pole(std::cos(angle), std::sin(angle));
        T lp_real = lp_pole.real();

        c10::complex<T> half_Bp = (B * lp_pole) / T(2);
        c10::complex<T> disc = half_Bp * half_Bp - omega_0_sq;
        c10::complex<T> sqrt_disc = std::sqrt(disc);
        c10::complex<T> bp1 = half_Bp + sqrt_disc;
        c10::complex<T> bp2 = half_Bp - sqrt_disc;

        poles[section_idx] = {bp1, lp_pole, sqrt_disc, half_Bp, true};
        a1_vals[section_idx] = -(bp1 + bp2).real();  // = -B * lp_real
        a2_vals[section_idx] = (bp1 * bp2).real();   // = omega_0²

        // For middle pole: a1 = -B*lp_real, a2 = omega_0²
        d_a1_d_w1[section_idx] = lp_real;   // d(-B*lp)/dw1 = lp
        d_a1_d_w2[section_idx] = -lp_real;  // d(-B*lp)/dw2 = -lp
        d_a2_d_w1[section_idx] = omega_p2;  // d(omega_0²)/dw1 = w2
        d_a2_d_w2[section_idx] = omega_p1;  // d(omega_0²)/dw2 = w1
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

    // Compute first derivatives of normalization factor (d_nf_d_w1, d_nf_d_w2)
    T d_log_gain_sq_d_w1 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w1;
    T d_log_gain_sq_d_w2 = T(4) * T(actual_n) / omega_0 * d_omega0_d_w2;

    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;

        T d_den_real_d_w1 = d_a2_d_w1[i] - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2[i] - T(2) * omega_0 * d_omega0_d_w2;

        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        d_log_gain_sq_d_w1 -= d_den_sq_d_w1 / den_sq;
        d_log_gain_sq_d_w2 -= d_den_sq_d_w2 / den_sq;
    }

    T d_total_gain_d_w1 = total_gain * d_log_gain_sq_d_w1 / T(2);
    T d_total_gain_d_w2 = total_gain * d_log_gain_sq_d_w2 / T(2);

    T d_nf_d_w1 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w1;
    T d_nf_d_w2 = (-nf / (T(actual_n) * total_gain)) * d_total_gain_d_w2;

    // =========================================================================
    // Compute grad_grad_sos: contributions from Jacobian terms
    // grad_grad_sos[i] = grad_grad_omega_p1 * J_1i + grad_grad_omega_p2 * J_2i
    // =========================================================================
    for (int64_t i = 0; i < actual_n; ++i) {
        T* gg_row = grad_grad_sos + i * 6;

        // b0 coefficient gets gradient from normalization factor
        gg_row[0] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_nf_d_w1 : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_nf_d_w2 : T(0));

        // b1, b2, a0 are constants (0, 0, 1) - no gradient
        gg_row[1] = T(0);
        gg_row[2] = T(0);
        gg_row[3] = T(0);

        // a1, a2 get gradients from pole derivatives
        gg_row[4] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_a1_d_w1[i] : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_a1_d_w2[i] : T(0));
        gg_row[5] = (has_grad_grad_omega_p1 ? grad_grad_omega_p1 * d_a2_d_w1[i] : T(0))
                  + (has_grad_grad_omega_p2 ? grad_grad_omega_p2 * d_a2_d_w2[i] : T(0));
    }

    // =========================================================================
    // Compute new_grad_omega_p1, new_grad_omega_p2 from Hessian terms
    // These require second derivatives of a1, a2, and nf
    // =========================================================================
    for (int64_t i = 0; i < actual_n; ++i) {
        const T* grad_row = grad_sos + i * 6;
        T grad_a1 = grad_row[4];
        T grad_a2 = grad_row[5];

        c10::complex<T> lp = poles[i].lp;
        c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
        c10::complex<T> half_Bp = poles[i].half_Bp;
        c10::complex<T> bp = poles[i].bp;
        bool is_bp1 = poles[i].is_bp1;
        bool is_middle = has_middle && (i == actual_n - 1);

        T d2_a1_d_w1_w1, d2_a1_d_w2_w2, d2_a1_d_w1_w2;
        T d2_a2_d_w1_w1, d2_a2_d_w2_w2, d2_a2_d_w1_w2;

        if (is_middle) {
            // For middle pole: a1 = -B*lp_real (linear in B), a2 = omega_0²
            // Second derivatives of a1 are all zero
            d2_a1_d_w1_w1 = T(0);
            d2_a1_d_w2_w2 = T(0);
            d2_a1_d_w1_w2 = T(0);

            // Second derivatives of a2 = omega_0² = w1*w2
            d2_a2_d_w1_w1 = T(0);
            d2_a2_d_w2_w2 = T(0);
            d2_a2_d_w1_w2 = T(1);
        } else {
            // Compute second derivatives of bandpass poles
            // D = half_Bp² - omega_0²
            // sqrt_D = sqrt(D)
            // bp = half_Bp ± sqrt_D

            // Second derivatives of D
            c10::complex<T> lp_sq = lp * lp;
            c10::complex<T> d2_D_d_w1_w1 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w2_w2 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w1_w2 = -lp_sq / T(2) - c10::complex<T>(T(1), T(0));

            // First derivatives of D (recompute for clarity)
            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

            c10::complex<T> D = sqrt_disc * sqrt_disc;

            // Second derivatives of sqrt_D using quotient rule:
            // d(sqrt_D)/dw = dD/dw / (2*sqrt_D)
            // d²(sqrt_D)/dw² = [d²D/dw² * 2*sqrt_D - dD/dw * dD/dw / sqrt_D] / (4*D)
            //                = [2*D * d²D/dw² - (dD/dw)²] / (4 * D^(3/2))

            c10::complex<T> D_cubed_half = D * sqrt_disc;  // D^(3/2)
            c10::complex<T> four_D_cubed_half = T(4) * D_cubed_half;

            // Avoid division by zero for small D
            T D_mag = std::abs(D);
            if (D_mag < std::numeric_limits<T>::epsilon() * T(100)) {
                // Near-zero discriminant: second derivatives are ill-defined
                // Use zero as a safe approximation
                d2_a1_d_w1_w1 = T(0);
                d2_a1_d_w2_w2 = T(0);
                d2_a1_d_w1_w2 = T(0);
                d2_a2_d_w1_w1 = T(0);
                d2_a2_d_w2_w2 = T(0);
                d2_a2_d_w1_w2 = T(0);
            } else {
                c10::complex<T> d2_sqrt_d_w1_w1 = (T(2) * D * d2_D_d_w1_w1 - d_D_d_w1 * d_D_d_w1) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w2_w2 = (T(2) * D * d2_D_d_w2_w2 - d_D_d_w2 * d_D_d_w2) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w1_w2 = (T(2) * D * d2_D_d_w1_w2 - d_D_d_w1 * d_D_d_w2) / four_D_cubed_half;

                // Second derivatives of bp = half_Bp ± sqrt_D
                // d²(half_Bp)/dw² = 0 for all combinations
                c10::complex<T> d2_bp_d_w1_w1, d2_bp_d_w2_w2, d2_bp_d_w1_w2;
                if (is_bp1) {
                    d2_bp_d_w1_w1 = d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = d2_sqrt_d_w1_w2;
                } else {
                    d2_bp_d_w1_w1 = -d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = -d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = -d2_sqrt_d_w1_w2;
                }

                // First derivatives of bp (recompute)
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

                // Second derivatives of a1 = -2 * Re(bp)
                d2_a1_d_w1_w1 = -T(2) * d2_bp_d_w1_w1.real();
                d2_a1_d_w2_w2 = -T(2) * d2_bp_d_w2_w2.real();
                d2_a1_d_w1_w2 = -T(2) * d2_bp_d_w1_w2.real();

                // Second derivatives of a2 = |bp|² = Re(bp)² + Im(bp)²
                // d(a2)/dw = 2 * Re(bp * conj(d_bp/dw)) = 2*(Re(bp)*Re(d_bp) + Im(bp)*Im(d_bp))
                // d²(a2)/dw² = 2 * [Re(d_bp/dw)² + Im(d_bp/dw)² + Re(bp)*Re(d²_bp/dw²) + Im(bp)*Im(d²_bp/dw²)]
                //            = 2 * [|d_bp/dw|² + Re(bp * conj(d²_bp/dw²))]

                T d2_a2_contrib_11 = d_bp_d_w1.real() * d_bp_d_w1.real() + d_bp_d_w1.imag() * d_bp_d_w1.imag()
                                   + bp.real() * d2_bp_d_w1_w1.real() + bp.imag() * d2_bp_d_w1_w1.imag();
                T d2_a2_contrib_22 = d_bp_d_w2.real() * d_bp_d_w2.real() + d_bp_d_w2.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w2_w2.real() + bp.imag() * d2_bp_d_w2_w2.imag();
                T d2_a2_contrib_12 = d_bp_d_w1.real() * d_bp_d_w2.real() + d_bp_d_w1.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w1_w2.real() + bp.imag() * d2_bp_d_w1_w2.imag();

                d2_a2_d_w1_w1 = T(2) * d2_a2_contrib_11;
                d2_a2_d_w2_w2 = T(2) * d2_a2_contrib_22;
                d2_a2_d_w1_w2 = T(2) * d2_a2_contrib_12;
            }
        }

        // Accumulate Hessian contributions
        if (has_grad_grad_omega_p1) {
            new_grad_omega_p1 += grad_a1 * d2_a1_d_w1_w1 * grad_grad_omega_p1;
            new_grad_omega_p1 += grad_a2 * d2_a2_d_w1_w1 * grad_grad_omega_p1;
            new_grad_omega_p2 += grad_a1 * d2_a1_d_w1_w2 * grad_grad_omega_p1;
            new_grad_omega_p2 += grad_a2 * d2_a2_d_w1_w2 * grad_grad_omega_p1;
        }
        if (has_grad_grad_omega_p2) {
            new_grad_omega_p1 += grad_a1 * d2_a1_d_w1_w2 * grad_grad_omega_p2;
            new_grad_omega_p1 += grad_a2 * d2_a2_d_w1_w2 * grad_grad_omega_p2;
            new_grad_omega_p2 += grad_a1 * d2_a1_d_w2_w2 * grad_grad_omega_p2;
            new_grad_omega_p2 += grad_a2 * d2_a2_d_w2_w2 * grad_grad_omega_p2;
        }
    }

    // Add Hessian contributions from normalization factor
    // nf = total_gain^(-1/n), where total_gain = sqrt(prod_i gain_i)
    // log(nf) = -1/(2n) * log(total_gain_sq)
    // log(total_gain_sq) = 4n*log(omega_0) - sum_i log(den_sq_i)

    T total_grad_b0 = T(0);
    for (int64_t i = 0; i < actual_n; ++i) {
        total_grad_b0 += grad_sos[i * 6];
    }

    // Second derivatives of log(omega_0)
    T d2_log_w0_d_w1_w1 = -d_omega0_d_w1 * d_omega0_d_w1 / omega_0_sq + d2_omega0_d_w1_w1 / omega_0;
    T d2_log_w0_d_w2_w2 = -d_omega0_d_w2 * d_omega0_d_w2 / omega_0_sq + d2_omega0_d_w2_w2 / omega_0;
    T d2_log_w0_d_w1_w2 = -d_omega0_d_w1 * d_omega0_d_w2 / omega_0_sq + d2_omega0_d_w1_w2 / omega_0;

    // Start with contribution from 4n*log(omega_0)
    T d2_log_gain_sq_d_w1_w1 = T(4) * T(actual_n) * d2_log_w0_d_w1_w1;
    T d2_log_gain_sq_d_w2_w2 = T(4) * T(actual_n) * d2_log_w0_d_w2_w2;
    T d2_log_gain_sq_d_w1_w2 = T(4) * T(actual_n) * d2_log_w0_d_w1_w2;

    // Subtract contributions from each log(den_sq_i)
    // For each section: den_sq = (a2 - omega_0²)² + a1² * omega_0²
    // d²log(den_sq)/dω² = d²(den_sq)/dω² / den_sq - (d(den_sq)/dω)² / den_sq²
    for (int64_t i = 0; i < actual_n; ++i) {
        T a1 = a1_vals[i];
        T a2 = a2_vals[i];
        T den_real = a2 - omega_0_sq;
        T den_sq = den_real * den_real + a1 * a1 * omega_0_sq;
        T den_sq_sq = den_sq * den_sq;

        // First derivatives of den_sq (already computed but recompute for clarity)
        T d_den_real_d_w1 = d_a2_d_w1[i] - T(2) * omega_0 * d_omega0_d_w1;
        T d_den_real_d_w2 = d_a2_d_w2[i] - T(2) * omega_0 * d_omega0_d_w2;

        T d_den_sq_d_w1 = T(2) * den_real * d_den_real_d_w1
                       + T(2) * a1 * omega_0_sq * d_a1_d_w1[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w1;
        T d_den_sq_d_w2 = T(2) * den_real * d_den_real_d_w2
                       + T(2) * a1 * omega_0_sq * d_a1_d_w2[i]
                       + T(2) * a1 * a1 * omega_0 * d_omega0_d_w2;

        // Second derivatives of den_real = a2 - omega_0²
        // d²(den_real)/dω² = d²a2/dω² - 2*(d_omega0/dω)² - 2*omega_0*d²_omega0/dω²
        bool is_middle = has_middle && (i == actual_n - 1);
        T d2_a1_d_w1_w1_i, d2_a1_d_w2_w2_i, d2_a1_d_w1_w2_i;
        T d2_a2_d_w1_w1_i, d2_a2_d_w2_w2_i, d2_a2_d_w1_w2_i;

        if (is_middle) {
            d2_a1_d_w1_w1_i = T(0); d2_a1_d_w2_w2_i = T(0); d2_a1_d_w1_w2_i = T(0);
            d2_a2_d_w1_w1_i = T(0); d2_a2_d_w2_w2_i = T(0); d2_a2_d_w1_w2_i = T(1);
        } else {
            // Use the Hessians computed earlier (need to recompute here)
            c10::complex<T> lp = poles[i].lp;
            c10::complex<T> sqrt_disc = poles[i].sqrt_disc;
            c10::complex<T> half_Bp = poles[i].half_Bp;
            c10::complex<T> bp = poles[i].bp;
            bool is_bp1 = poles[i].is_bp1;

            c10::complex<T> lp_sq = lp * lp;
            c10::complex<T> d2_D_d_w1_w1 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w2_w2 = lp_sq / T(2);
            c10::complex<T> d2_D_d_w1_w2 = -lp_sq / T(2) - c10::complex<T>(T(1), T(0));

            c10::complex<T> d_half_Bp_d_w1 = -lp / T(2);
            c10::complex<T> d_half_Bp_d_w2 = lp / T(2);
            c10::complex<T> d_D_d_w1 = T(2) * half_Bp * d_half_Bp_d_w1 - c10::complex<T>(omega_p2, T(0));
            c10::complex<T> d_D_d_w2 = T(2) * half_Bp * d_half_Bp_d_w2 - c10::complex<T>(omega_p1, T(0));

            c10::complex<T> D = sqrt_disc * sqrt_disc;
            T D_mag = std::abs(D);

            if (D_mag < std::numeric_limits<T>::epsilon() * T(100)) {
                d2_a1_d_w1_w1_i = T(0); d2_a1_d_w2_w2_i = T(0); d2_a1_d_w1_w2_i = T(0);
                d2_a2_d_w1_w1_i = T(0); d2_a2_d_w2_w2_i = T(0); d2_a2_d_w1_w2_i = T(0);
            } else {
                c10::complex<T> D_cubed_half = D * sqrt_disc;
                c10::complex<T> four_D_cubed_half = T(4) * D_cubed_half;

                c10::complex<T> d2_sqrt_d_w1_w1 = (T(2) * D * d2_D_d_w1_w1 - d_D_d_w1 * d_D_d_w1) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w2_w2 = (T(2) * D * d2_D_d_w2_w2 - d_D_d_w2 * d_D_d_w2) / four_D_cubed_half;
                c10::complex<T> d2_sqrt_d_w1_w2 = (T(2) * D * d2_D_d_w1_w2 - d_D_d_w1 * d_D_d_w2) / four_D_cubed_half;

                c10::complex<T> d2_bp_d_w1_w1, d2_bp_d_w2_w2, d2_bp_d_w1_w2;
                if (is_bp1) {
                    d2_bp_d_w1_w1 = d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = d2_sqrt_d_w1_w2;
                } else {
                    d2_bp_d_w1_w1 = -d2_sqrt_d_w1_w1;
                    d2_bp_d_w2_w2 = -d2_sqrt_d_w2_w2;
                    d2_bp_d_w1_w2 = -d2_sqrt_d_w1_w2;
                }

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

                d2_a1_d_w1_w1_i = -T(2) * d2_bp_d_w1_w1.real();
                d2_a1_d_w2_w2_i = -T(2) * d2_bp_d_w2_w2.real();
                d2_a1_d_w1_w2_i = -T(2) * d2_bp_d_w1_w2.real();

                T d2_a2_contrib_11 = d_bp_d_w1.real() * d_bp_d_w1.real() + d_bp_d_w1.imag() * d_bp_d_w1.imag()
                                   + bp.real() * d2_bp_d_w1_w1.real() + bp.imag() * d2_bp_d_w1_w1.imag();
                T d2_a2_contrib_22 = d_bp_d_w2.real() * d_bp_d_w2.real() + d_bp_d_w2.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w2_w2.real() + bp.imag() * d2_bp_d_w2_w2.imag();
                T d2_a2_contrib_12 = d_bp_d_w1.real() * d_bp_d_w2.real() + d_bp_d_w1.imag() * d_bp_d_w2.imag()
                                   + bp.real() * d2_bp_d_w1_w2.real() + bp.imag() * d2_bp_d_w1_w2.imag();

                d2_a2_d_w1_w1_i = T(2) * d2_a2_contrib_11;
                d2_a2_d_w2_w2_i = T(2) * d2_a2_contrib_22;
                d2_a2_d_w1_w2_i = T(2) * d2_a2_contrib_12;
            }
        }

        // Second derivatives of den_real = a2 - omega_0²
        T d2_den_real_d_w1_w1 = d2_a2_d_w1_w1_i - T(2) * d_omega0_d_w1 * d_omega0_d_w1 - T(2) * omega_0 * d2_omega0_d_w1_w1;
        T d2_den_real_d_w2_w2 = d2_a2_d_w2_w2_i - T(2) * d_omega0_d_w2 * d_omega0_d_w2 - T(2) * omega_0 * d2_omega0_d_w2_w2;
        T d2_den_real_d_w1_w2 = d2_a2_d_w1_w2_i - T(2) * d_omega0_d_w1 * d_omega0_d_w2 - T(2) * omega_0 * d2_omega0_d_w1_w2;

        // Second derivatives of den_sq = den_real² + a1² * omega_0²
        // d²(den_sq)/dω² = d²[(a2-ω₀²)² + a₁²ω₀²]/dω²
        // The a₁²ω₀² term expands to: 2(da₁/dω)²ω₀² + 2a₁(d²a₁/dω²)ω₀² + 8a₁(da₁/dω)ω₀(dω₀/dω)
        //                           + 2a₁²(dω₀/dω)² + 2a₁²ω₀(d²ω₀/dω²)
        // Note: coefficient 8 comes from product rule on a₁²ω₀² giving 4 terms with factor 2
        T d2_den_sq_d_w1_w1 = T(2) * d_den_real_d_w1 * d_den_real_d_w1 + T(2) * den_real * d2_den_real_d_w1_w1
                           + T(2) * d_a1_d_w1[i] * d_a1_d_w1[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w1_w1_i
                           + T(8) * a1 * omega_0 * d_a1_d_w1[i] * d_omega0_d_w1
                           + T(2) * a1 * a1 * d_omega0_d_w1 * d_omega0_d_w1 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w1_w1;

        T d2_den_sq_d_w2_w2 = T(2) * d_den_real_d_w2 * d_den_real_d_w2 + T(2) * den_real * d2_den_real_d_w2_w2
                           + T(2) * d_a1_d_w2[i] * d_a1_d_w2[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w2_w2_i
                           + T(8) * a1 * omega_0 * d_a1_d_w2[i] * d_omega0_d_w2
                           + T(2) * a1 * a1 * d_omega0_d_w2 * d_omega0_d_w2 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w2_w2;

        T d2_den_sq_d_w1_w2 = T(2) * d_den_real_d_w1 * d_den_real_d_w2 + T(2) * den_real * d2_den_real_d_w1_w2
                           + T(2) * d_a1_d_w1[i] * d_a1_d_w2[i] * omega_0_sq + T(2) * a1 * omega_0_sq * d2_a1_d_w1_w2_i
                           + T(4) * a1 * omega_0 * (d_a1_d_w1[i] * d_omega0_d_w2 + d_a1_d_w2[i] * d_omega0_d_w1)
                           + T(2) * a1 * a1 * d_omega0_d_w1 * d_omega0_d_w2 + T(2) * a1 * a1 * omega_0 * d2_omega0_d_w1_w2;

        // d²log(den_sq)/dω² = d²(den_sq)/dω² / den_sq - (d(den_sq)/dω)² / den_sq²
        T d2_log_den_sq_d_w1_w1 = d2_den_sq_d_w1_w1 / den_sq - d_den_sq_d_w1 * d_den_sq_d_w1 / den_sq_sq;
        T d2_log_den_sq_d_w2_w2 = d2_den_sq_d_w2_w2 / den_sq - d_den_sq_d_w2 * d_den_sq_d_w2 / den_sq_sq;
        T d2_log_den_sq_d_w1_w2 = d2_den_sq_d_w1_w2 / den_sq - d_den_sq_d_w1 * d_den_sq_d_w2 / den_sq_sq;

        // Subtract from log(total_gain_sq) Hessian
        d2_log_gain_sq_d_w1_w1 -= d2_log_den_sq_d_w1_w1;
        d2_log_gain_sq_d_w2_w2 -= d2_log_den_sq_d_w2_w2;
        d2_log_gain_sq_d_w1_w2 -= d2_log_den_sq_d_w1_w2;
    }

    // Convert log(total_gain_sq) Hessian to nf Hessian
    // log(nf) = -1/(2n) * log(total_gain_sq)
    // nf = exp(log_nf)
    // d²nf/dω² = nf * [(d log_nf/dω)² + d² log_nf/dω²]
    T factor = T(-1) / (T(2) * T(actual_n));
    T d_log_nf_d_w1 = factor * d_log_gain_sq_d_w1;
    T d_log_nf_d_w2 = factor * d_log_gain_sq_d_w2;

    T d2_log_nf_d_w1_w1 = factor * d2_log_gain_sq_d_w1_w1;
    T d2_log_nf_d_w2_w2 = factor * d2_log_gain_sq_d_w2_w2;
    T d2_log_nf_d_w1_w2 = factor * d2_log_gain_sq_d_w1_w2;

    T d2_nf_d_w1_w1 = nf * (d_log_nf_d_w1 * d_log_nf_d_w1 + d2_log_nf_d_w1_w1);
    T d2_nf_d_w2_w2 = nf * (d_log_nf_d_w2 * d_log_nf_d_w2 + d2_log_nf_d_w2_w2);
    T d2_nf_d_w1_w2 = nf * (d_log_nf_d_w1 * d_log_nf_d_w2 + d2_log_nf_d_w1_w2);

    if (has_grad_grad_omega_p1) {
        new_grad_omega_p1 += total_grad_b0 * d2_nf_d_w1_w1 * grad_grad_omega_p1;
        new_grad_omega_p2 += total_grad_b0 * d2_nf_d_w1_w2 * grad_grad_omega_p1;
    }
    if (has_grad_grad_omega_p2) {
        new_grad_omega_p1 += total_grad_b0 * d2_nf_d_w1_w2 * grad_grad_omega_p2;
        new_grad_omega_p2 += total_grad_b0 * d2_nf_d_w2_w2 * grad_grad_omega_p2;
    }
}

}  // namespace torchscience::impl::filter
