#pragma once

#include <ATen/ATen.h>

// Forward declarations for the example operator kernels
// These functions are implemented in backend-specific files and can be
// tested directly in C++ unit tests

namespace science {
namespace ops {

// CPU kernels (implemented in ops/cpu/example_kernel.cpp)
namespace cpu {
    at::Tensor example_forward_kernel(const at::Tensor& input, const at::Scalar& x);
    at::Tensor example_backward_kernel(const at::Tensor& grad_out, const at::Tensor& input,
                                       const at::Scalar& x);
}  // namespace cpu

#ifdef WITH_CUDA
// CUDA kernels (implemented in ops/cuda/example_kernel.cu)
namespace cuda {
    at::Tensor example_forward_kernel(const at::Tensor& input, const at::Scalar& x);
    at::Tensor example_backward_kernel(const at::Tensor& grad_out, const at::Tensor& input,
                                       const at::Scalar& x);

    // CUDA helper functions exposed for testing
    int get_num_threads();
    int get_num_blocks(int64_t numel, int threads_per_block);
}  // namespace cuda
#endif

// Meta kernels for shape inference (implemented in ops/meta/example_kernel.cpp)
// These are already exposed in the namespace
at::Tensor example_forward_meta(const at::Tensor& input, const at::Scalar& x);
at::Tensor example_backward_meta(const at::Tensor& grad_out, const at::Tensor& input,
                                 const at::Scalar& x);

}  // namespace ops
}  // namespace science
