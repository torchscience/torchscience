#pragma once

#include <torchscience/csrc/impl/special_functions/mobius_mu.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t mobius_mu(scalar_t x) {
  return torchscience::impl::special_functions::mobius_mu(x);
}

template <typename scalar_t>
__device__ scalar_t mobius_mu_backward(scalar_t x) {
  return torchscience::impl::special_functions::mobius_mu_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(mobius_mu)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(mobius_mu)

} // namespace torchscience::cuda::special_functions
