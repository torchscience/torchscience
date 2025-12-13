#pragma once

#include <torchscience/csrc/impl/special_functions/factorial.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t factorial(scalar_t x) {
  return torchscience::impl::special_functions::factorial(x);
}

template <typename scalar_t>
__device__ scalar_t factorial_backward(scalar_t x) {
  return torchscience::impl::special_functions::factorial_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(factorial)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(factorial)

} // namespace torchscience::cuda::special_functions
