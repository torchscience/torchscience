#pragma once

#include <torchscience/csrc/impl/special_functions/jacobi_elliptic_sd.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t jacobi_elliptic_sd(scalar_t u, scalar_t k) {
  return torchscience::impl::special_functions::jacobi_elliptic_sd(u, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> jacobi_elliptic_sd_backward(scalar_t u, scalar_t k) {
  auto [grad_u, grad_k] = torchscience::impl::special_functions::jacobi_elliptic_sd_backward(u, k);
  return std::make_tuple(grad_u, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(jacobi_elliptic_sd, u, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(jacobi_elliptic_sd)

} // namespace torchscience::cuda::special_functions
