#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../impl/integral_transform/inverse_hilbert_transform.h"
#include "../../impl/integral_transform/inverse_hilbert_transform_backward.h"
#include "../../impl/integral_transform/inverse_hilbert_transform_backward_backward.h"

namespace torchscience::cuda::integral_transform {

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * CUDA kernel to apply inverse Hilbert frequency response in-place.
 *
 * Multiplies each frequency bin by h_inv[k] = i * sign(freq[k]):
 *   - k=0 (DC): multiply by 0
 *   - k=1...(n-1)/2 (positive freq): multiply by +i
 *   - k=n/2 (Nyquist, even n): multiply by 0
 *   - k=(n+1)/2...n-1 (negative freq): multiply by -i
 *
 * @param data Complex spectrum data (batch_size * n elements)
 * @param n Signal length (number of frequency bins)
 * @param batch_size Number of signals in batch
 */
template <typename scalar_t>
__global__ void apply_inverse_hilbert_response_kernel(
    c10::complex<scalar_t>* __restrict__ data,
    int64_t n,
    int64_t batch_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = n * batch_size;

    if (idx >= total_elements) return;

    int64_t freq_idx = idx % n;

    // Compute h_inv[freq_idx] = i * sign(freq) = -h[freq_idx]
    c10::complex<scalar_t> h;

    if (freq_idx == 0) {
        // DC component: h_inv[0] = 0
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (n % 2 == 0 && freq_idx == n / 2) {
        // Nyquist for even n: h_inv[n/2] = 0
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (freq_idx < (n + 1) / 2) {
        // Positive frequencies: h_inv[k] = +i (opposite of forward Hilbert)
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(1));
    } else {
        // Negative frequencies: h_inv[k] = -i (opposite of forward Hilbert)
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(-1));
    }

    data[idx] = data[idx] * h;
}

/**
 * CUDA kernel to apply backward inverse Hilbert frequency response in-place.
 *
 * For backward pass of H^{-1}, we need (H^{-1})^T = H.
 * So use forward Hilbert response: h[k] = -i * sign(freq[k])
 */
template <typename scalar_t>
__global__ void apply_inverse_hilbert_backward_response_kernel(
    c10::complex<scalar_t>* __restrict__ data,
    int64_t n,
    int64_t batch_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = n * batch_size;

    if (idx >= total_elements) return;

    int64_t freq_idx = idx % n;

    c10::complex<scalar_t> h;

    if (freq_idx == 0) {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (n % 2 == 0 && freq_idx == n / 2) {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (freq_idx < (n + 1) / 2) {
        // Backward of H^{-1} is H: -i for positive frequencies
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(-1));
    } else {
        // Backward of H^{-1} is H: +i for negative frequencies
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(1));
    }

    data[idx] = data[idx] * h;
}

}  // namespace

/**
 * CUDA implementation of inverse Hilbert transform.
 *
 * Uses cuFFT (via ATen) for FFT/IFFT and custom CUDA kernel for
 * applying the inverse Hilbert frequency response.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @return Inverse Hilbert transform of the input
 */
at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
) {
    TORCH_CHECK(input.is_cuda(), "inverse_hilbert_transform: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "inverse_hilbert_transform: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "inverse_hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "inverse_hilbert_transform: transform dimension must have positive size");

    // Determine FFT length: use n_param if specified, otherwise input size
    int64_t n = (n_param > 0) ? n_param : input.size(dim);
    TORCH_CHECK(n > 0, "inverse_hilbert_transform: n must be positive");

    at::Tensor input_contig = input.contiguous();

    // Compute FFT along specified dimension (with optional padding/truncation via n)
    c10::optional<int64_t> fft_n = (n_param > 0) ? c10::optional<int64_t>(n) : c10::nullopt;
    at::Tensor spectrum = at::fft_fft(input_contig, fft_n, dim);

    // Calculate batch size (product of all dims except transform dim)
    int64_t batch_size = spectrum.numel() / n;

    // Move transform dimension to last for efficient kernel access
    // Using movedim for consistency with hilbert_transform implementation
    at::Tensor spectrum_transposed = spectrum.movedim(dim, -1).contiguous();

    // Apply inverse Hilbert frequency response using custom CUDA kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t total_elements = batch_size * n;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(
        at::toRealValueType(spectrum_transposed.scalar_type()),
        "inverse_hilbert_transform_cuda_kernel",
        [&]() {
            auto* data = reinterpret_cast<c10::complex<scalar_t>*>(
                spectrum_transposed.data_ptr()
            );

            apply_inverse_hilbert_response_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                data, n, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    // Move transform dimension back to original position
    at::Tensor modified_spectrum = spectrum_transposed.movedim(-1, dim).contiguous();

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part
    if (!input.is_complex()) {
        result = at::real(result);
    }

    return result;
}

/**
 * CUDA backward pass for inverse Hilbert transform.
 *
 * The forward operation is: output = H^{-1}_n[pad_n(input)] or H^{-1}_n[trunc_n(input)]
 * where pad_n zero-pads or trunc_n truncates input to size n along dim.
 *
 * The adjoint of zero-padding is truncation (keeping first input_size elements).
 * The adjoint of truncation is zero-padding (padding with zeros to input_size).
 * The adjoint of H^{-1} is H (since (H^{-1})^T = (-H)^T = -H^T = -(-H) = H).
 *
 * So the backward is:
 * - Apply H_n to grad_output (produces size n)
 * - If n > input_size: truncate to input_size (adjoint of padding)
 * - If n < input_size: zero-pad to input_size (adjoint of truncation)
 */
at::Tensor inverse_hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_hilbert_transform_backward: grad_output must be a CUDA tensor");

    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply H (which equals -H^{-1}) with size n to grad_output
    // Since (H^{-1})^T = H = -H^{-1}, we compute: grad = -inverse_hilbert_transform(grad_output)
    at::Tensor grad = -inverse_hilbert_transform(grad_output, n_param, dim);

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
 * CUDA double backward pass for inverse Hilbert transform.
 *
 * The backward computes grad_x = H[grad_y] = -H^{-1}[grad_y].
 * The Jacobian of this mapping wrt grad_y is -H^{-1}.
 * The transpose of -H^{-1} is -(H^{-1})^T = -H.
 *
 * Since H = -H^{-1}, we have: -H = -(-H^{-1}) = H^{-1}.
 * So grad_grad_output = H^{-1}[grad_grad_input] = inverse_hilbert_transform(grad_grad_input).
 * new_grad_input = 0 since H^{-1} is linear (no second-order terms).
 */
std::tuple<at::Tensor, at::Tensor> inverse_hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim
) {
    // grad_grad_output: gradient with respect to grad_output
    // This is (-H^{-1})^T applied to grad_grad_input = -H[grad_grad_input] = H^{-1}[grad_grad_input]
    at::Tensor grad_grad_output = inverse_hilbert_transform(grad_grad_input, n_param, dim);

    // new_grad_input = 0 (H^{-1} is linear, no second-order term)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform
    );

    module.impl(
        "inverse_hilbert_transform_backward",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform_backward
    );

    module.impl(
        "inverse_hilbert_transform_backward_backward",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform_backward_backward
    );
}
