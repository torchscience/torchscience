#pragma once

#include <torchscience/csrc/impl/special_functions/log_gamma.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t log_gamma(scalar_t x) {
  return torchscience::impl::special_functions::log_gamma(x);
}

template <typename scalar_t>
__device__ scalar_t log_gamma_backward(scalar_t x) {
  return torchscience::impl::special_functions::log_gamma_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(log_gamma)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(log_gamma)

} // namespace torchscience::cuda::special_functions
