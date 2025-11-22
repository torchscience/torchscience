#include "../example.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science {
namespace ops {

// Meta kernel for example operator
// This is used by PyTorch's FakeTensor system to propagate metadata
// (sizes, strides, dtype, device) without actually executing the kernel
at::Tensor example_forward_meta(const at::Tensor& input, const at::Scalar& x) {
    // Return a tensor with the same properties as input
    // The output has the same shape, dtype, and device as the input
    return at::empty_like(input);
}

// Meta kernel for example backward
at::Tensor example_backward_meta(const at::Tensor& grad_out, const at::Tensor& input,
                                 const at::Scalar& x) {
    // The gradient has the same shape as grad_out (which has the same shape as input)
    return at::empty_like(grad_out);
}

}  // namespace ops
}  // namespace science

// Register Meta implementations for PT2 (torch.compile) support
TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("example", science::ops::example_forward_meta);
    module.impl("_example_backward", science::ops::example_backward_meta);
}
