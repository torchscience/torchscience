#pragma once

/*
 * Hilbert Transform Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * The Hilbert transform of a function f(t) is defined as:
 *
 *   H[f](x) = (1/pi) * PV integral from -inf to inf of f(t)/(t-x) dt
 *
 * where PV denotes the Cauchy principal value.
 *
 * DISCRETE IMPLEMENTATION:
 * ========================
 * For discrete signals, the Hilbert transform is computed via FFT:
 *
 *   1. Compute FFT: F[k] = FFT(f)
 *   2. Multiply by frequency response: H[k] = F[k] * h[k]
 *      where h[k] = -i * sign(freq[k])
 *        - h[0] = 0 (DC component)
 *        - h[k] = -i for positive frequencies (k = 1, ..., N/2-1)
 *        - h[N/2] = 0 for even N (Nyquist)
 *        - h[k] = +i for negative frequencies (k = N/2+1, ..., N-1)
 *   3. Compute IFFT: H[f] = IFFT(H)
 *
 * KEY PROPERTIES:
 * ===============
 * 1. H[sin(wt)] = -cos(wt), H[cos(wt)] = sin(wt)
 * 2. H[H[f]] = -f (involutory up to sign)
 * 3. Energy preservation: integral |H[f]|^2 = integral |f|^2
 * 4. Linearity: H[af + bg] = aH[f] + bH[g]
 * 5. Inverse: H^{-1}[f] = -H[f]
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/pad.h>
#include <cmath>
#include <vector>

namespace torchscience::impl::integral_transform {

/**
 * Padding mode enum matching PyTorch conventions.
 * Maps to torch.nn.functional.pad modes.
 */
enum class PaddingMode : int64_t {
    Constant = 0,
    Reflect = 1,
    Replicate = 2,
    Circular = 3
};

/**
 * Convert padding mode integer to string for ATen pad function.
 */
inline std::string padding_mode_to_string(int64_t mode) {
    switch (static_cast<PaddingMode>(mode)) {
        case PaddingMode::Constant: return "constant";
        case PaddingMode::Reflect: return "reflect";
        case PaddingMode::Replicate: return "replicate";
        case PaddingMode::Circular: return "circular";
        default:
            TORCH_CHECK(false, "Invalid padding_mode: ", mode,
                ". Must be 0 (constant), 1 (reflect), 2 (replicate), or 3 (circular)");
    }
}

/**
 * Apply a single step of padding to a tensor (internal helper).
 * Handles the 2D requirement for non-constant padding modes.
 *
 * @param input Input tensor (last dimension is the one to pad)
 * @param pad_amount Amount to pad on the right
 * @param padding_mode Padding mode
 * @param padding_value Value for constant padding
 * @param needs_unsqueeze Whether we need to add a batch dimension
 * @return Padded tensor
 */
inline at::Tensor apply_padding_step(
    const at::Tensor& input,
    int64_t pad_amount,
    int64_t padding_mode,
    double padding_value,
    bool needs_unsqueeze
) {
    at::Tensor input_work = input;

    if (needs_unsqueeze) {
        input_work = input_work.unsqueeze(0);
    }

    std::vector<int64_t> pad_sizes = {0, pad_amount};
    std::string mode_str = padding_mode_to_string(padding_mode);

    at::Tensor padded;
    if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
        padded = at::pad(input_work, pad_sizes, mode_str, padding_value);
    } else {
        padded = at::pad(input_work, pad_sizes, mode_str);
    }

    if (needs_unsqueeze) {
        padded = padded.squeeze(0);
    }

    return padded;
}

/**
 * Apply padding to tensor along specified dimension.
 *
 * For non-constant padding modes, if the padding amount exceeds the input size,
 * padding is applied iteratively in chunks to satisfy PyTorch's constraints.
 *
 * @param input Input tensor
 * @param target_size Target size along dim after padding
 * @param dim Dimension to pad
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @return Padded tensor
 */
inline at::Tensor apply_padding(
    const at::Tensor& input,
    int64_t target_size,
    int64_t dim,
    int64_t padding_mode,
    double padding_value
) {
    int64_t current_size = input.size(dim);

    if (target_size <= current_size) {
        // No padding needed (truncation handled elsewhere)
        return input;
    }

    int64_t total_pad = target_size - current_size;

    // Move target dim to last position for F.pad
    at::Tensor result = input.movedim(dim, -1);

    // PyTorch's at::pad requires at least 2D input for non-constant padding modes.
    bool needs_unsqueeze = (result.dim() == 1) &&
        (padding_mode != static_cast<int64_t>(PaddingMode::Constant));

    // For constant padding, we can do it all at once
    if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
        result = apply_padding_step(result, total_pad, padding_mode, padding_value, needs_unsqueeze);
    } else {
        // For reflect mode, padding must be < current size
        // For replicate/circular, padding can be any size in recent PyTorch,
        // but we handle all non-constant modes the same way for robustness.
        // We iteratively pad in chunks when the padding is too large.
        int64_t remaining_pad = total_pad;

        while (remaining_pad > 0) {
            int64_t current_dim_size = result.size(-1);
            // For reflect: pad < size, so max_pad = size - 1
            // We use size - 1 to be safe for all non-constant modes
            int64_t max_pad = current_dim_size - 1;
            if (max_pad <= 0) {
                // Should not happen with valid input, but handle gracefully
                max_pad = 1;
            }

            int64_t pad_this_step = std::min(remaining_pad, max_pad);
            result = apply_padding_step(result, pad_this_step, padding_mode, padding_value, needs_unsqueeze);
            remaining_pad -= pad_this_step;
        }
    }

    // Move dimension back
    return result.movedim(-1, dim);
}

/**
 * Apply window function to tensor along specified dimension.
 *
 * @param input Input tensor
 * @param window 1-D window tensor (must match input size along dim)
 * @param dim Dimension to apply window
 * @return Windowed tensor
 */
inline at::Tensor apply_window(
    const at::Tensor& input,
    const at::Tensor& window,
    int64_t dim
) {
    TORCH_CHECK(window.dim() == 1,
        "window must be 1-D, got ", window.dim(), "-D tensor");
    TORCH_CHECK(window.size(0) == input.size(dim),
        "window size (", window.size(0), ") must match input size along dim (",
        input.size(dim), ")");

    // Reshape window for broadcasting: (1, 1, ..., n, ..., 1)
    std::vector<int64_t> window_shape(input.dim(), 1);
    window_shape[dim] = window.size(0);

    at::Tensor window_reshaped = window.view(window_shape);

    return input * window_reshaped;
}

/**
 * Compute the Hilbert transform frequency response multiplier for index k.
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier h[k] = -i * sign(freq[k])
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> hilbert_frequency_response(int64_t k, int64_t n) {
    // DC component: h[0] = 0
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // Nyquist frequency for even-length signals: h[n/2] = 0
    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    // Positive frequencies: h[k] = -i (standard Hilbert transform convention)
    // k = 1, 2, ..., floor((n-1)/2)
    // h = -i * sign(freq) = -i * 1 = -i for positive frequencies
    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(-1));
    }

    // Negative frequencies: h[k] = +i
    // h = -i * sign(freq) = -i * (-1) = +i for negative frequencies
    // k = ceil((n+1)/2), ..., n-1
    return c10::complex<T>(T(0), T(1));
}

/**
 * Compute sign factor for Hilbert transform.
 *
 * Returns:
 *  0 for DC and Nyquist
 * -1 for positive frequencies (h = -i means imaginary part is -1)
 * +1 for negative frequencies (h = +i means imaginary part is +1)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T hilbert_sign_factor(int64_t k, int64_t n) {
    if (k == 0) {
        return T(0);
    }
    if (n % 2 == 0 && k == n / 2) {
        return T(0);
    }
    if (k < (n + 1) / 2) {
        return T(-1);
    }
    return T(1);
}

/**
 * Apply Hilbert frequency response in-place to a complex spectrum.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_frequency_response<T>(k, n);
        spectrum[k] = spectrum[k] * h;
    }
}

/**
 * Apply Hilbert frequency response, storing result in separate output.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_frequency_response<T>(k, n);
        output[k] = input[k] * h;
    }
}

/**
 * Compute the analytic signal from real input.
 *
 * The analytic signal is z(t) = f(t) + i*H[f](t)
 * where H[f] is the Hilbert transform.
 *
 * In frequency domain: Z[k] = F[k] * (1 + sign(freq[k]))
 *   - Z[0] = F[0] (DC preserved)
 *   - Z[k] = 2*F[k] for positive frequencies
 *   - Z[k] = 0 for negative frequencies
 *   - Z[n/2] = F[n/2] for even n (Nyquist preserved)
 *
 * @param k Frequency index
 * @param n Total number of frequency bins
 * @return Scale factor for analytic signal computation
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T analytic_signal_scale(int64_t k, int64_t n) {
    if (k == 0) {
        return T(1);  // DC
    }
    if (n % 2 == 0 && k == n / 2) {
        return T(1);  // Nyquist
    }
    if (k < (n + 1) / 2) {
        return T(2);  // Positive frequencies
    }
    return T(0);  // Negative frequencies
}

}  // namespace torchscience::impl::integral_transform
