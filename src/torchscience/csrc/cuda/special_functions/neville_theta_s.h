#pragma once

#include <torchscience/csrc/impl/special_functions/neville_theta_s.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t neville_theta_s(scalar_t k, scalar_t u) {
  return torchscience::impl::special_functions::neville_theta_s(k, u);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> neville_theta_s_backward(scalar_t k, scalar_t u) {
  auto [grad_k, grad_u] = torchscience::impl::special_functions::neville_theta_s_backward(k, u);
  return std::make_tuple(grad_k, grad_u);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(neville_theta_s, k, u)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(neville_theta_s)

} // namespace torchscience::cuda::special_functions
