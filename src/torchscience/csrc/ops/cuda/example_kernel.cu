#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace {

// CUDA kernel for the example operator (adds scalar to all elements)
// This demonstrates a simple element-wise GPU kernel
template <typename scalar_t>
__global__ void example_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel,
    scalar_t x
) {
    // Calculate global thread ID
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop to handle cases where numel > num_threads
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        // Add scalar to each element: output = input + x
        output[i] = input[i] + x;
    }
}

// Helper to get optimal CUDA launch configuration
inline int get_num_threads() {
    return 256;
}

inline int get_num_blocks(int64_t numel, int threads_per_block) {
    const int max_blocks = 65535;
    return std::min(max_blocks, static_cast<int>((numel + threads_per_block - 1) / threads_per_block));
}

// Forward pass: adds scalar x to all elements with actual CUDA kernel execution
at::Tensor example_forward_kernel(
    const at::Tensor& input,
    const at::Scalar& x
) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");

    at::cuda::CUDAGuard device_guard(input.device());

    // Create output tensor with same properties as input
    auto output = at::empty_like(input);

    int64_t numel = input.numel();

    if (numel == 0) {
        return output;
    }

    // Get CUDA launch configuration
    const int threads = get_num_threads();
    const int blocks = get_num_blocks(numel, threads);

    // Dispatch kernel based on tensor dtype
    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        input.scalar_type(),
        "example_cuda_kernel",
        [&] {
            example_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel,
                x.to<scalar_t>()
            );
        }
    );

    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

// Backward pass: gradient with respect to input
at::Tensor example_backward_kernel(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Scalar& x
) {
    // Unused parameters
    (void)input;
    (void)x;

    at::cuda::CUDAGuard device_guard(grad_out.device());

    // Gradient of (input + x) with respect to input is 1
    // So gradient just passes through unchanged
    return grad_out.contiguous();
}

} // namespace

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        TORCH_SELECTIVE_NAME("torchscience::example"),
        TORCH_FN(example_forward_kernel)
    );

    module.impl(
        TORCH_SELECTIVE_NAME("torchscience::_example_backward"),
        TORCH_FN(example_backward_kernel)
    );
}

} // namespace ops
} // namespace science
