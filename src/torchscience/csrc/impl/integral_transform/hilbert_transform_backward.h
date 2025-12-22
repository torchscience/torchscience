#pragma once

/*
 * Hilbert Transform Backward Implementation
 *
 * GRADIENT DERIVATION:
 * ====================
 * The Hilbert transform is a linear operator, so its gradient is straightforward.
 *
 * Forward: y = H[x]
 * Backward: grad_x = -H[grad_y]
 *
 * This is because H is orthogonal (preserves inner products):
 *   <H[f], g> = <f, H^T[g]> = <f, -H[g]>
 *
 * For the gradient of a loss L with respect to input x:
 *   dL/dx = H^T[dL/dy] = -H[dL/dy]
 *
 * For real-valued functions:
 *   H^T = -H (the adjoint of H is -H)
 *
 * Therefore: grad_input = -H[grad_output]
 *
 * However, in practice with FFT implementation:
 *   H[f] = IFFT(FFT(f) * h)
 *   H^T[f] = IFFT(FFT(f) * conj(h)) = IFFT(FFT(f) * (-h)) = -H[f]
 *
 * Therefore: grad_input = -H[grad_output] = H^{-1}[grad_output]
 *
 * Since H[H[f]] = -f, we have H^{-1} = -H, confirming:
 *   grad_input = -H[grad_output]
 */

#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include "hilbert_transform.h"

namespace torchscience::impl::integral_transform {

/**
 * Compute the backward frequency response multiplier.
 *
 * For backward pass, we need the adjoint of H, which is -H.
 * So the frequency response is negated: h_backward[k] = -h[k] = i * sign(freq[k])
 *
 * @param k Frequency index (0 to n-1)
 * @param n Total number of frequency bins
 * @return Complex multiplier for backward pass
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> hilbert_backward_frequency_response(int64_t k, int64_t n) {
    // DC component: still 0
    if (k == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // Nyquist frequency for even-length signals: still 0
    if (n % 2 == 0 && k == n / 2) {
        return c10::complex<T>(T(0), T(0));
    }

    // Positive frequencies: h_backward[k] = +i (negated from forward -i)
    if (k < (n + 1) / 2) {
        return c10::complex<T>(T(0), T(1));
    }

    // Negative frequencies: h_backward[k] = -i (negated from forward +i)
    return c10::complex<T>(T(0), T(-1));
}

/**
 * Apply backward Hilbert frequency response in-place.
 *
 * This applies -H (the adjoint of H) in the frequency domain.
 *
 * @param spectrum Complex FFT output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_response_inplace(c10::complex<T>* spectrum, int64_t n) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_backward_frequency_response<T>(k, n);
        spectrum[k] = spectrum[k] * h;
    }
}

/**
 * Apply backward Hilbert frequency response to spectrum.
 *
 * @param input Complex FFT input array of length n
 * @param output Complex output array of length n
 * @param n Number of frequency bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void apply_hilbert_backward_response(
    const c10::complex<T>* input,
    c10::complex<T>* output,
    int64_t n
) {
    for (int64_t k = 0; k < n; ++k) {
        c10::complex<T> h = hilbert_backward_frequency_response<T>(k, n);
        output[k] = input[k] * h;
    }
}

/**
 * Compute backward for one step of reflect padding.
 *
 * When forward did: output = pad(input, [0, pad_amount], "reflect")
 * The backward accumulates gradients from the reflected positions.
 *
 * @param grad_output Gradient w.r.t. padded output (along last dim)
 * @param input_size Original input size
 * @param pad_amount Amount that was padded on the right
 * @return Gradient w.r.t. input before padding
 */
inline at::Tensor reflect_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    // Start with gradients from the non-padded region
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    // Accumulate gradients from the reflected region
    // Reflect padding mirrors around the last element (excluding it)
    // padded[input_size + i] = input[input_size - 2 - i] for i = 0, 1, ..., pad_amount-1
    for (int64_t i = 0; i < pad_amount; ++i) {
        int64_t src_pos = input_size + i;       // Position in padded tensor
        int64_t dst_pos = input_size - 2 - i;   // Corresponding position in input

        // Handle wrap-around for large padding (iterative reflect)
        while (dst_pos < 0) {
            dst_pos = -dst_pos;
        }
        while (dst_pos >= input_size) {
            dst_pos = 2 * input_size - 2 - dst_pos;
            if (dst_pos < 0) dst_pos = -dst_pos;
        }

        if (dst_pos >= 0 && dst_pos < input_size) {
            grad_input.select(-1, dst_pos).add_(grad_output.select(-1, src_pos));
        }
    }

    return grad_input;
}

/**
 * Compute backward for one step of replicate padding.
 *
 * @param grad_output Gradient w.r.t. padded output
 * @param input_size Original input size
 * @param pad_amount Amount that was padded on the right
 * @return Gradient w.r.t. input before padding
 */
inline at::Tensor replicate_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    // Replicate copies the last element to all padded positions
    // All gradients from padded region accumulate to the last element
    for (int64_t i = 0; i < pad_amount; ++i) {
        grad_input.select(-1, input_size - 1).add_(
            grad_output.select(-1, input_size + i)
        );
    }

    return grad_input;
}

/**
 * Compute backward for one step of circular padding.
 *
 * @param grad_output Gradient w.r.t. padded output
 * @param input_size Original input size
 * @param pad_amount Amount that was padded on the right
 * @return Gradient w.r.t. input before padding
 */
inline at::Tensor circular_padding_backward_step(
    const at::Tensor& grad_output,
    int64_t input_size,
    int64_t pad_amount
) {
    at::Tensor grad_input = grad_output.narrow(-1, 0, input_size).clone();

    // Circular padding wraps around: padded[input_size + i] = input[i % input_size]
    for (int64_t i = 0; i < pad_amount; ++i) {
        int64_t dst_pos = i % input_size;
        grad_input.select(-1, dst_pos).add_(
            grad_output.select(-1, input_size + i)
        );
    }

    return grad_input;
}

/**
 * Adjust gradient size to match input shape, accounting for padding mode.
 *
 * Handles the size adjustment for the backward pass when n != input_size.
 * For non-constant padding modes, properly accumulates gradients from
 * padded regions back to the original input positions.
 *
 * @param grad The gradient tensor (after applying -H)
 * @param input_size The original input size along the transform dimension
 * @param n The FFT length used in forward pass
 * @param dim The normalized dimension (must be non-negative)
 * @param padding_mode Padding mode used in forward pass
 * @return Adjusted gradient tensor matching original input size
 */
inline at::Tensor adjust_backward_gradient_size(
    const at::Tensor& grad,
    int64_t input_size,
    int64_t n,
    int64_t dim,
    int64_t padding_mode = 0
) {
    if (n > input_size) {
        // Move target dim to last for processing
        at::Tensor grad_moved = grad.movedim(dim, -1).contiguous();

        if (padding_mode == static_cast<int64_t>(PaddingMode::Constant)) {
            // Constant padding: just truncate (padded values don't depend on input)
            at::Tensor result = grad_moved.narrow(-1, 0, input_size).contiguous();
            return result.movedim(-1, dim);
        }

        // For non-constant modes, we need to reverse the iterative padding
        // The forward did iterative padding in chunks of max_pad = current_size - 1
        // We need to reverse this: start from size n and work back to input_size

        // Reconstruct the sizes used in forward iterative padding
        std::vector<int64_t> sizes;
        int64_t current = input_size;
        while (current < n) {
            sizes.push_back(current);
            int64_t max_pad = current - 1;
            current = std::min(current + max_pad, n);
        }
        // sizes now contains [input_size, size_after_step1, size_after_step2, ...]
        // We need to reverse the padding from n back to input_size

        at::Tensor grad_work = grad_moved;

        // Reverse iterate through the padding steps
        for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
            int64_t target_size = *it;
            int64_t current_size = grad_work.size(-1);
            int64_t pad_amount = current_size - target_size;

            if (pad_amount > 0) {
                if (padding_mode == static_cast<int64_t>(PaddingMode::Reflect)) {
                    grad_work = reflect_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(PaddingMode::Replicate)) {
                    grad_work = replicate_padding_backward_step(grad_work, target_size, pad_amount);
                } else if (padding_mode == static_cast<int64_t>(PaddingMode::Circular)) {
                    grad_work = circular_padding_backward_step(grad_work, target_size, pad_amount);
                }
            }
        }

        return grad_work.movedim(-1, dim);
    } else if (n < input_size) {
        // Zero-pad to input_size along dim (adjoint of truncation)
        std::vector<int64_t> pad_shape(grad.sizes().begin(), grad.sizes().end());
        pad_shape[dim] = input_size;
        at::Tensor padded = at::zeros(pad_shape, grad.options());
        padded.narrow(dim, 0, n).copy_(grad);
        return padded;
    }
    return grad.contiguous();
}

}  // namespace torchscience::impl::integral_transform
