#include <ATen/ATen.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace {

// Adds scalar x to all elements for quantized tensors
at::Tensor example_forward_kernel(
    const at::Tensor& input,
    const at::Scalar& x
) {
    TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
    TORCH_CHECK(input.is_quantized(), "input must be a quantized tensor");

    // For quantized tensors, we dequantize, add, and re-quantize
    // This preserves the quantization parameters
    auto dequantized = input.dequantize();
    auto result = dequantized + x;
    return at::quantize_per_tensor(result, input.q_scale(), input.q_zero_point(), input.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchscience, QuantizedCPU, module) {
    module.impl(
        TORCH_SELECTIVE_NAME("torchscience::example"),
        TORCH_FN(example_forward_kernel)
    );
}

} // namespace ops
} // namespace science
