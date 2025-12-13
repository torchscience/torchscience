#pragma once

#include <torchscience/csrc/impl/special_functions/airy_ai_derivative.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t airy_ai_derivative(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai_derivative(x);
}

template <typename scalar_t>
__device__ scalar_t airy_ai_derivative_backward(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai_derivative_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(airy_ai_derivative)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(airy_ai_derivative)

} // namespace torchscience::cuda::special_functions
