#pragma once

#include <torchscience/csrc/impl/special_functions/airy_bi_derivative.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t airy_bi_derivative(scalar_t x) {
  return torchscience::impl::special_functions::airy_bi_derivative(x);
}

template <typename scalar_t>
__device__ scalar_t airy_bi_derivative_backward(scalar_t x) {
  return torchscience::impl::special_functions::airy_bi_derivative_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(airy_bi_derivative)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(airy_bi_derivative)

} // namespace torchscience::cuda::special_functions
