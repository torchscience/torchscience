#include "../../example.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace science {
namespace ops {

// Forward kernel for sparse CUDA tensors
// For sparse tensors, we add the scalar to the sparse values only
at::Tensor example_forward_kernel(const at::Tensor& input, const at::Scalar& x) {
    TORCH_CHECK(input.is_sparse(), "Input must be a sparse tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    // Ensure we're on the right device
    c10::cuda::CUDAGuard device_guard(input.device());

    // For sparse tensors, we only need to add the scalar to the values
    // The indices remain unchanged
    auto values = input._values();
    auto new_values = values + x;

    // Create new sparse tensor with same indices but updated values
    // Use sparse_coo_tensor instead of _unsafe to preserve autograd
    return at::sparse_coo_tensor(input._indices(), new_values, input.sizes(), input.options());
}

// Backward kernel for sparse CUDA tensors
at::Tensor example_backward_kernel(const at::Tensor& grad_out, const at::Tensor& input,
                                   const at::Scalar& x) {
    TORCH_CHECK(grad_out.is_sparse(), "Gradient must be a sparse tensor");
    TORCH_CHECK(input.is_sparse(), "Input must be a sparse tensor");
    TORCH_CHECK(grad_out.is_cuda(), "Gradient must be a CUDA tensor");

    // Ensure we're on the right device
    c10::cuda::CUDAGuard device_guard(grad_out.device());

    // For addition, gradient passes through unchanged
    // Just ensure it's coalesced
    return grad_out.coalesce();
}

}  // namespace ops
}  // namespace science

// Register SparseCUDA implementation
TORCH_LIBRARY_IMPL(torchscience, SparseCUDA, module) {
    module.impl("example", science::ops::example_forward_kernel);
    module.impl("_example_backward", science::ops::example_backward_kernel);
}
