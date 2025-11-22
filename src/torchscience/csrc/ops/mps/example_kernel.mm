#include <ATen/ATen.h>

// Suppress availability warnings for MPS APIs
// The MPS backend requires macOS 12.0+ at runtime
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

namespace science {
namespace ops {

namespace {

// Embedded Metal shader library
static at::native::mps::MetalShaderLibrary lib(R"SCIENCE_METAL(

#include <metal_stdlib>
using namespace metal;

// Helper macro for 1D kernel loops
#define MPS_1D_KERNEL_LOOP(i, n, n_tgs)                 \
  for (uint i = (tgid.x * tptg.x) + tid2.x; i < (n);   \
       i += (tptg.x * n_tgs))

// Example kernel - adds scalar to all elements
template<typename T>
kernel void example_kernel(
    constant T*       input     [[buffer(0)]],
    device T*         output    [[buffer(1)]],
    constant int64_t& numel     [[buffer(2)]],
    constant T&       x         [[buffer(3)]],
    uint3             tgid      [[threadgroup_position_in_grid]],
    uint3             tptg      [[threads_per_threadgroup]],
    uint3             tid2      [[thread_position_in_threadgroup]]
) {
    MPS_1D_KERNEL_LOOP(index, numel, 1) {
        // Add scalar to each element: output = input + x
        output[index] = input[index] + x;
    }
}

// Register kernel variants for different data types
#define REGISTER_EXAMPLE_OP(DTYPE)                              \
template                                                        \
[[host_name("example_" #DTYPE)]]                                \
kernel void example_kernel<DTYPE>(                              \
    constant DTYPE*       input     [[buffer(0)]],              \
    device DTYPE*         output    [[buffer(1)]],              \
    constant int64_t&     numel     [[buffer(2)]],              \
    constant DTYPE&       x         [[buffer(3)]],              \
    uint3                 tgid      [[threadgroup_position_in_grid]], \
    uint3                 tptg      [[threads_per_threadgroup]], \
    uint3                 tid2      [[thread_position_in_threadgroup]]);

// Register for common types
REGISTER_EXAMPLE_OP(float);
REGISTER_EXAMPLE_OP(half);
REGISTER_EXAMPLE_OP(int);
REGISTER_EXAMPLE_OP(long);

)SCIENCE_METAL");

// Helper function to get pipeline state for a kernel
static id<MTLComputePipelineState> sciencePipelineState(
    id<MTLDevice> device,
    const std::string& kernel) {
  return lib.getPipelineStateForFunc(kernel);
}


at::Tensor example_forward_kernel(
    const at::Tensor& input,
    const at::Scalar& x
) {
    using namespace at::native::mps;

    TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");

    // Make input contiguous for Metal buffer access
    at::Tensor input_c = input.contiguous();

    at::DeviceGuard guard(input_c.device());

    // Create output tensor
    auto output = at::empty_like(input_c);

    int64_t numel = input_c.numel();

    if (numel == 0) {
        return output;
    }

    // Get Metal buffers
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_c);
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);

    // Get Metal device and kernel
    id<MTLDevice> device = MPSDevice::getInstance()->device();

    // Get kernel name based on dtype
    std::string kernelName = "example_" + scalarToMetalTypeString(input.scalar_type());

    // Get compiled pipeline state
    id<MTLComputePipelineState> pipelineState = sciencePipelineState(device, kernelName);

    if (pipelineState == nil) {
        TORCH_CHECK(false, "Failed to get pipeline state for kernel: ", kernelName);
    }

    // Calculate threadgroup configuration
    NSUInteger threadsPerThreadgroup = pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger threadgroups = (numel + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

    MTLSize threadGroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    MTLSize threadgroupsPerGrid = MTLSizeMake(threadgroups, 1, 1);

    // Get MPS stream and execute
    MPSStream* mpsStream = getCurrentMPSStream();

    dispatch_sync(mpsStream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

            [computeEncoder setComputePipelineState:pipelineState];

            // Set all arguments - need to convert scalar to the right type for Metal
            AT_DISPATCH_ALL_TYPES_AND2(
                at::kHalf, at::kBFloat16,
                input.scalar_type(),
                "example_mps_setargs",
                [&] {
                    scalar_t x_val = x.to<scalar_t>();
                    mtl_setArgs(computeEncoder, inputBuffer, outputBuffer, numel, x_val);
                }
            );

            [computeEncoder dispatchThreadgroups:threadgroupsPerGrid
                           threadsPerThreadgroup:threadGroupSize];
        }
    });

    return output;
}

at::Tensor example_backward_kernel(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Scalar& x
) {
    // Unused parameters
    (void)input;
    (void)x;

    // Gradient of (input + x) with respect to input is 1
    // So gradient just passes through unchanged
    return grad_out.contiguous();
}

} // namespace

TORCH_LIBRARY_IMPL(torchscience, MPS, module) {
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

#pragma clang diagnostic pop
