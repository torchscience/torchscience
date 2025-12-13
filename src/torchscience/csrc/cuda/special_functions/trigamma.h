#pragma once

#include <torchscience/csrc/impl/special_functions/trigamma.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t trigamma(scalar_t x) {
  return torchscience::impl::special_functions::trigamma(x);
}

template <typename scalar_t>
__device__ scalar_t trigamma_backward(scalar_t x) {
  return torchscience::impl::special_functions::trigamma_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(trigamma)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(trigamma)

} // namespace torchscience::cuda::special_functions
