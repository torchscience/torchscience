#pragma once

#include <torchscience/csrc/impl/special_functions/whittaker_w.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t whittaker_w(scalar_t kappa, scalar_t mu, scalar_t z) {
  return torchscience::impl::special_functions::whittaker_w(kappa, mu, z);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> whittaker_w_backward(scalar_t kappa, scalar_t mu, scalar_t z) {
  auto [grad_kappa, grad_mu, grad_z] = torchscience::impl::special_functions::whittaker_w_backward(kappa, mu, z);
  return std::make_tuple(grad_kappa, grad_mu, grad_z);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(whittaker_w, kappa, mu, z)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(whittaker_w)

} // namespace torchscience::cuda::special_functions
