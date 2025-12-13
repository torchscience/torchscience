#pragma once

#include <torchscience/csrc/impl/special_functions/double_factorial.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t double_factorial(scalar_t x) {
  return torchscience::impl::special_functions::double_factorial(x);
}

template <typename scalar_t>
__device__ scalar_t double_factorial_backward(scalar_t x) {
  return torchscience::impl::special_functions::double_factorial_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(double_factorial)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(double_factorial)

} // namespace torchscience::cuda::special_functions
