#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../impl/filter/butterworth_analog_bandpass_filter.h"

namespace torchscience::cuda::filter {

/**
 * CUDA kernel for butterworth_analog_bandpass_filter.
 *
 * Each thread computes one filter (one batch element).
 */
template <typename scalar_t>
__global__ void butterworth_analog_bandpass_filter_kernel(
    const scalar_t* __restrict__ omega_p1,
    const scalar_t* __restrict__ omega_p2,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        scalar_t w1 = omega_p1[idx];
        scalar_t w2 = omega_p2[idx];
        scalar_t* sos_ptr = output + idx * n * 6;

        impl::filter::butterworth_analog_bandpass_filter<scalar_t>(
            n, w1, w2, sos_ptr
        );
    }
}

/**
 * CUDA implementation of butterworth_analog_bandpass_filter.
 */
inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    TORCH_CHECK(n > 0, "butterworth_analog_bandpass_filter: order n must be positive, got ", n);
    TORCH_CHECK(n <= 64, "butterworth_analog_bandpass_filter: order n must be <= 64, got ", n);

    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(omega_p1.device());

    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    // Get batch shape
    auto batch_shape = omega_p1_bc.sizes().vec();

    // Compute output shape: (*batch_shape, n, 6)
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(n);
    output_shape.push_back(6);

    // Create output tensor
    auto options = omega_p1_bc.options();
    at::Tensor output = at::empty(output_shape, options);

    // Flatten batch dimensions for kernel launch
    int64_t batch_size = omega_p1_bc.numel();
    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();
    at::Tensor output_flat = output.view({batch_size, n, 6});

    // Launch kernel
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_cuda",
        [&]() {
            butterworth_analog_bandpass_filter_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                omega_p1_flat.data_ptr<scalar_t>(),
                omega_p2_flat.data_ptr<scalar_t>(),
                output_flat.data_ptr<scalar_t>(),
                batch_size,
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return output;
}

/**
 * CUDA backward kernel for butterworth_analog_bandpass_filter.
 */
template <typename scalar_t>
__global__ void butterworth_analog_bandpass_filter_backward_kernel(
    const scalar_t* __restrict__ grad_sos,
    const scalar_t* __restrict__ omega_p1,
    const scalar_t* __restrict__ omega_p2,
    scalar_t* __restrict__ grad_omega_p1,
    scalar_t* __restrict__ grad_omega_p2,
    int64_t batch_size,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        scalar_t w1 = omega_p1[idx];
        scalar_t w2 = omega_p2[idx];
        const scalar_t* grad_sos_ptr = grad_sos + idx * n * 6;

        scalar_t grad_w1, grad_w2;
        impl::filter::butterworth_analog_bandpass_filter_backward<scalar_t>(
            grad_sos_ptr, n, w1, w2, grad_w1, grad_w2
        );

        grad_omega_p1[idx] = grad_w1;
        grad_omega_p2[idx] = grad_w2;
    }
}

/**
 * CUDA backward implementation.
 */
inline std::tuple<at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward(
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    c10::cuda::CUDAGuard device_guard(omega_p1.device());

    // Broadcast omega_p1 and omega_p2 to common shape
    auto broadcasted = at::broadcast_tensors({omega_p1, omega_p2});
    at::Tensor omega_p1_bc = broadcasted[0].contiguous();
    at::Tensor omega_p2_bc = broadcasted[1].contiguous();

    auto batch_shape = omega_p1_bc.sizes().vec();
    int64_t batch_size = omega_p1_bc.numel();

    at::Tensor omega_p1_flat = omega_p1_bc.flatten();
    at::Tensor omega_p2_flat = omega_p2_bc.flatten();
    at::Tensor grad_output_flat = grad_output.view({batch_size, n, 6}).contiguous();

    at::Tensor grad_omega_p1_flat = at::empty({batch_size}, omega_p1_flat.options());
    at::Tensor grad_omega_p2_flat = at::empty({batch_size}, omega_p2_flat.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        omega_p1_flat.scalar_type(),
        "butterworth_analog_bandpass_filter_backward_cuda",
        [&]() {
            butterworth_analog_bandpass_filter_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output_flat.data_ptr<scalar_t>(),
                omega_p1_flat.data_ptr<scalar_t>(),
                omega_p2_flat.data_ptr<scalar_t>(),
                grad_omega_p1_flat.data_ptr<scalar_t>(),
                grad_omega_p2_flat.data_ptr<scalar_t>(),
                batch_size,
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    at::Tensor grad_omega_p1 = grad_omega_p1_flat.view(batch_shape);
    at::Tensor grad_omega_p2 = grad_omega_p2_flat.view(batch_shape);

    return std::make_tuple(grad_omega_p1, grad_omega_p2);
}

/**
 * CUDA double-backward implementation.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> butterworth_analog_bandpass_filter_backward_backward(
    const at::Tensor& grad_grad_omega_p1,
    const at::Tensor& grad_grad_omega_p2,
    const at::Tensor& grad_output,
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    // For now, return zeros for second-order gradients
    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor grad_omega_p1 = at::zeros_like(omega_p1);
    at::Tensor grad_omega_p2 = at::zeros_like(omega_p2);

    return std::make_tuple(grad_grad_output, grad_omega_p1, grad_omega_p2);
}

}  // namespace torchscience::cuda::filter

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::cuda::filter::butterworth_analog_bandpass_filter
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward",
        &torchscience::cuda::filter::butterworth_analog_bandpass_filter_backward
    );

    module.impl(
        "butterworth_analog_bandpass_filter_backward_backward",
        &torchscience::cuda::filter::butterworth_analog_bandpass_filter_backward_backward
    );
}
