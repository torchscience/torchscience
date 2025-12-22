#pragma once

#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

#include "../../impl/integral_transform/hilbert_transform.h"
#include "../../impl/integral_transform/hilbert_transform_backward.h"
#include "../../impl/integral_transform/hilbert_transform_backward_backward.h"

namespace torchscience::cpu::integral_transform {

/**
 * CPU implementation of Hilbert transform with padding and windowing.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @param window Optional window tensor to apply before FFT
 * @return Hilbert transform of the input
 */
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.numel() > 0, "hilbert_transform: input tensor must be non-empty");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "hilbert_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "hilbert_transform: n must be positive");

    // Ensure contiguous for efficient operations
    at::Tensor processed = input.contiguous();

    // Apply padding if needed
    if (n > input_size) {
        processed = impl::integral_transform::apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        // Truncation
        processed = processed.narrow(dim, 0, n);
    }

    // Apply window if provided
    if (window.has_value()) {
        processed = impl::integral_transform::apply_window(
            processed, window.value(), dim
        );
    }

    // Compute FFT along specified dimension
    at::Tensor spectrum = at::fft_fft(processed, c10::nullopt, dim);

    // Create frequency response tensor
    std::vector<int64_t> response_shape(ndim, 1);
    response_shape[dim] = n;

    at::Tensor response = at::zeros(response_shape, spectrum.options());

    // Fill in the response values
    AT_DISPATCH_COMPLEX_TYPES(
        spectrum.scalar_type(),
        "hilbert_transform_cpu_response",
        [&]() {
            using real_t = typename c10::scalar_value_type<scalar_t>::type;

            auto response_flat = response.view({n});
            auto* response_data = response_flat.data_ptr<scalar_t>();

            for (int64_t k = 0; k < n; ++k) {
                auto h = impl::integral_transform::hilbert_frequency_response<real_t>(k, n);
                response_data[k] = scalar_t(h.real(), h.imag());
            }
        }
    );

    // Apply frequency response
    at::Tensor modified_spectrum = spectrum * response;

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part
    if (!input.is_complex()) {
        result = at::real(result).contiguous();
    }

    return result;
}

/**
 * Backward pass for Hilbert transform on CPU.
 */
inline at::Tensor hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H with size n to grad_output
    at::Tensor grad = -hilbert_transform(grad_output, n_param, dim, 0, 0.0, c10::nullopt);

    // If window was applied in forward, multiply gradient by window BEFORE size adjustment
    // (chain rule: d/dx[w*x] = w * d/dx[x], where w operates at padded size)
    if (window.has_value()) {
        grad = impl::integral_transform::apply_window(grad, window.value(), norm_dim);
    }

    // Adjust size to match input shape (with proper gradient accumulation for padding)
    grad = impl::integral_transform::adjust_backward_gradient_size(
        grad, input_size, n, norm_dim, padding_mode
    );

    return grad;
}

/**
 * Double backward pass for Hilbert transform on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // grad_grad_output = H[grad_grad_input]
    at::Tensor grad_grad_output = hilbert_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window
    );

    // No second-order term (H is linear)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cpu::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "hilbert_transform",
        &torchscience::cpu::integral_transform::hilbert_transform
    );

    module.impl(
        "hilbert_transform_backward",
        &torchscience::cpu::integral_transform::hilbert_transform_backward
    );

    module.impl(
        "hilbert_transform_backward_backward",
        &torchscience::cpu::integral_transform::hilbert_transform_backward_backward
    );
}
