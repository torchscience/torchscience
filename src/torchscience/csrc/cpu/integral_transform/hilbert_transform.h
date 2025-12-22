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

namespace {

/**
 * Create the Hilbert frequency response tensor.
 *
 * @param n Signal length
 * @param options Tensor options (dtype, device)
 * @return Complex tensor of shape (n,) containing frequency response
 */
inline at::Tensor create_hilbert_response(int64_t n, const at::TensorOptions& options) {
    auto response = at::zeros({n}, options.dtype(at::kComplexFloat));

    AT_DISPATCH_FLOATING_TYPES(
        options.dtype().toScalarType(),
        "create_hilbert_response",
        [&]() {
            auto* data = response.data_ptr<c10::complex<float>>();

            at::parallel_for(0, n, 0, [&](int64_t begin, int64_t end) {
                for (int64_t k = begin; k < end; ++k) {
                    data[k] = impl::integral_transform::hilbert_frequency_response<float>(k, n);
                }
            });
        }
    );

    return response;
}

/**
 * Create the backward Hilbert frequency response tensor.
 */
inline at::Tensor create_hilbert_backward_response(int64_t n, const at::TensorOptions& options) {
    auto response = at::zeros({n}, options.dtype(at::kComplexFloat));

    auto* data = response.data_ptr<c10::complex<float>>();

    at::parallel_for(0, n, 0, [&](int64_t begin, int64_t end) {
        for (int64_t k = begin; k < end; ++k) {
            data[k] = impl::integral_transform::hilbert_backward_frequency_response<float>(k, n);
        }
    });

    return response;
}

}  // namespace

/**
 * CPU implementation of Hilbert transform.
 *
 * Computes the Hilbert transform along a specified dimension using FFT.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @return Hilbert transform of the input
 */
inline at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
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

    // Determine FFT length: use n_param if specified, otherwise input size
    int64_t n = (n_param > 0) ? n_param : input.size(dim);
    TORCH_CHECK(n > 0, "hilbert_transform: n must be positive");

    // Ensure contiguous for efficient FFT
    at::Tensor input_contig = input.contiguous();

    // Compute FFT along specified dimension (with optional padding/truncation via n)
    c10::optional<int64_t> fft_n = (n_param > 0) ? c10::optional<int64_t>(n) : c10::nullopt;
    at::Tensor spectrum = at::fft_fft(input_contig, fft_n, dim);

    // Create frequency response tensor
    // Shape needs to broadcast: (1, 1, ..., n, ..., 1, 1) with n at position dim
    std::vector<int64_t> response_shape(ndim, 1);
    response_shape[dim] = n;

    at::Tensor response = at::zeros(response_shape, spectrum.options());

    // Fill in the response values
    AT_DISPATCH_COMPLEX_TYPES(
        spectrum.scalar_type(),
        "hilbert_transform_cpu_response",
        [&]() {
            using real_t = typename c10::scalar_value_type<scalar_t>::type;

            // Get 1D view for filling
            auto response_flat = response.view({n});
            auto* response_data = response_flat.data_ptr<scalar_t>();

            for (int64_t k = 0; k < n; ++k) {
                auto h = impl::integral_transform::hilbert_frequency_response<real_t>(k, n);
                response_data[k] = scalar_t(h.real(), h.imag());
            }
        }
    );

    // Apply frequency response (element-wise multiplication with broadcasting)
    at::Tensor modified_spectrum = spectrum * response;

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part (as contiguous tensor)
    if (!input.is_complex()) {
        result = at::real(result).contiguous();
    }

    return result;
}

/**
 * Backward pass for Hilbert transform on CPU.
 *
 * The forward operation is: output = H_n[pad_n(input)] or H_n[trunc_n(input)]
 * where pad_n zero-pads or trunc_n truncates input to size n along dim.
 *
 * The adjoint of zero-padding is truncation (keeping first input_size elements).
 * The adjoint of truncation is zero-padding (padding with zeros to input_size).
 * The adjoint of H is -H (since H^T = -H).
 *
 * So the backward is:
 * - Apply -H_n to grad_output (produces size n)
 * - If n > input_size: truncate to input_size (adjoint of padding)
 * - If n < input_size: zero-pad to input_size (adjoint of truncation)
 */
inline at::Tensor hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H with size n to grad_output
    at::Tensor grad = -hilbert_transform(grad_output, n_param, dim);

    // Adjust size to match input shape
    if (n > input_size) {
        // Truncate: take first input_size elements along dim
        grad = grad.narrow(norm_dim, 0, input_size);
    } else if (n < input_size) {
        // Zero-pad to input_size along dim
        std::vector<int64_t> pad_shape(grad.sizes().begin(), grad.sizes().end());
        pad_shape[norm_dim] = input_size;
        at::Tensor padded = at::zeros(pad_shape, grad.options());
        padded.narrow(norm_dim, 0, n).copy_(grad);
        grad = padded;
    }

    return grad.contiguous();
}

/**
 * Double backward pass for Hilbert transform on CPU.
 *
 * The backward computes grad_x = -H[grad_y].
 * The Jacobian of this mapping wrt grad_y is -H.
 * The transpose of -H is -H^T = -(-H) = H.
 *
 * So grad_grad_output = H[grad_grad_input].
 * new_grad_input = 0 since H is linear (no second-order terms).
 */
inline std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
) {
    // grad_grad_output: gradient with respect to grad_output
    // This is (-H)^T applied to grad_grad_input = H[grad_grad_input]
    at::Tensor grad_grad_output = hilbert_transform(grad_grad_input, n_param, dim);

    // new_grad_input: additional gradient contribution to input
    // Since H is linear, there's no second-order term, so this is zero
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
