#include "../example.h"

#ifdef USE_ROCM

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <torch/library.h>

// HIP kernel implementation
// HIP is AMD's equivalent to CUDA for GPU programming
namespace science {
namespace ops {

// HIP kernel for element-wise addition
template <typename scalar_t>
__global__ void example_hip_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output, int64_t numel, scalar_t x) {
    // Grid-stride loop pattern
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        output[i] = input[i] + x;
    }
}

// Forward pass for HIP
at::Tensor example_forward_kernel(const at::Tensor& input, const at::Scalar& x) {
    TORCH_CHECK(input.is_hip(), "Input must be a HIP tensor");

    // Ensure we're on the right device
    c10::hip::HIPGuard device_guard(input.device());

    // Allocate output tensor
    auto output = at::empty_like(input);

    // Get tensor properties
    auto numel = input.numel();
    if (numel == 0) {
        return output;
    }

    // Launch configuration
    // Use 256 threads per block (optimal for most AMD GPUs)
    const int threads = 256;
    const int blocks = std::min((numel + threads - 1) / threads, (int64_t)65535);

    // Get current HIP stream
    auto stream = at::hip::getCurrentHIPStream();

    // Dispatch based on data type
    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "example_hip_kernel", [&] {
            example_hip_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel, x.to<scalar_t>());
        });

    // Check for kernel launch errors
    C10_HIP_KERNEL_LAUNCH_CHECK();

    return output;
}

// Backward pass for HIP
at::Tensor example_backward_kernel(const at::Tensor& grad_out, const at::Tensor& input,
                                   const at::Scalar& x) {
    TORCH_CHECK(grad_out.is_hip(), "Gradient must be a HIP tensor");

    // For addition, gradient passes through unchanged
    return grad_out.contiguous();
}

}  // namespace ops
}  // namespace science

// Register HIP implementation
TORCH_LIBRARY_IMPL(torchscience, HIP, module) {
    module.impl("example", science::ops::example_forward_kernel);
    module.impl("_example_backward", science::ops::example_backward_kernel);
}

#endif  // USE_ROCM
