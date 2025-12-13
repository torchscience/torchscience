#pragma once

#include <torchscience/csrc/impl/special_functions/whittaker_m.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t whittaker_m(scalar_t kappa, scalar_t mu, scalar_t z) {
  return torchscience::impl::special_functions::whittaker_m(kappa, mu, z);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> whittaker_m_backward(scalar_t kappa, scalar_t mu, scalar_t z) {
  auto [grad_kappa, grad_mu, grad_z] = torchscience::impl::special_functions::whittaker_m_backward(kappa, mu, z);
  return std::make_tuple(grad_kappa, grad_mu, grad_z);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(whittaker_m, kappa, mu, z)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(whittaker_m)

} // namespace torchscience::cuda::special_functions
