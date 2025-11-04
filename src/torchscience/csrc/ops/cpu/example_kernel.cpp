#include <ATen/ATen.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace {

// Forward pass: adds scalar x to all elements of input
at::Tensor example_forward_kernel(
    const at::Tensor& input,
    const at::Scalar& x
) {
    TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

    // Add scalar to all elements: output = input + x
    return input + x;
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

    // Gradient of (input + x) with respect to input is 1
    // So gradient just passes through unchanged
    return grad_out.contiguous();
}

} // namespace

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
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
