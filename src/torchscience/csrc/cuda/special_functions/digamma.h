#pragma once

#include <torchscience/csrc/impl/special_functions/digamma.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t digamma(scalar_t x) {
  return torchscience::impl::special_functions::digamma(x);
}

template <typename scalar_t>
__device__ scalar_t digamma_backward(scalar_t x) {
  return torchscience::impl::special_functions::digamma_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(digamma)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(digamma)

} // namespace torchscience::cuda::special_functions
