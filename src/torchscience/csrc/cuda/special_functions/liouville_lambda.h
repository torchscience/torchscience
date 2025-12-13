#pragma once

#include <torchscience/csrc/impl/special_functions/liouville_lambda.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t liouville_lambda(scalar_t x) {
  return torchscience::impl::special_functions::liouville_lambda(x);
}

template <typename scalar_t>
__device__ scalar_t liouville_lambda_backward(scalar_t x) {
  return torchscience::impl::special_functions::liouville_lambda_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(liouville_lambda)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(liouville_lambda)

} // namespace torchscience::cuda::special_functions
