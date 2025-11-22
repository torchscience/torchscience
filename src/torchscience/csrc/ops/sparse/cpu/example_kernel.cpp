#include "../../example.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace {

// Forward kernel for sparse CPU tensors
// For sparse tensors, we add the scalar to the sparse values only
at::Tensor example_forward_kernel(const at::Tensor& input, const at::Scalar& x) {
    TORCH_CHECK(input.is_sparse(), "Input must be a sparse tensor");
    TORCH_CHECK(input.device().is_cpu(), "Input must be a CPU tensor");

    // For sparse tensors, we only need to add the scalar to the values
    // The indices remain unchanged
    auto values = input._values();
    auto new_values = values + x;

    // Create new sparse tensor with same indices but updated values
    // Use sparse_coo_tensor instead of _unsafe to preserve autograd
    return at::sparse_coo_tensor(input._indices(), new_values, input.sizes(), input.options());
}

// Backward kernel for sparse CPU tensors
at::Tensor example_backward_kernel(const at::Tensor& grad_out, const at::Tensor& input,
                                   const at::Scalar& x) {
    TORCH_CHECK(grad_out.is_sparse(), "Gradient must be a sparse tensor");
    TORCH_CHECK(input.is_sparse(), "Input must be a sparse tensor");

    // For addition, gradient passes through unchanged
    // Just ensure it's contiguous
    return grad_out.coalesce();
}

}  // namespace
}  // namespace ops
}  // namespace science

// Register SparseCPU implementation
TORCH_LIBRARY_IMPL(torchscience, SparseCPU, module) {
    module.impl("example", TORCH_FN(example_forward_kernel));
    module.impl("_example_backward", TORCH_FN(example_backward_kernel));
}
