#pragma once

#include <torchscience/csrc/impl/special_functions/gamma.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t gamma(scalar_t x) {
  return torchscience::impl::special_functions::gamma(x);
}

template <typename scalar_t>
__device__ scalar_t gamma_backward(scalar_t x) {
  return torchscience::impl::special_functions::gamma_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(gamma)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(gamma)

} // namespace torchscience::cuda::special_functions
