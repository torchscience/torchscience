#pragma once

#include <torchscience/csrc/impl/special_functions/jacobi_elliptic_sc.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t jacobi_elliptic_sc(scalar_t u, scalar_t k) {
  return torchscience::impl::special_functions::jacobi_elliptic_sc(u, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> jacobi_elliptic_sc_backward(scalar_t u, scalar_t k) {
  auto [grad_u, grad_k] = torchscience::impl::special_functions::jacobi_elliptic_sc_backward(u, k);
  return std::make_tuple(grad_u, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(jacobi_elliptic_sc, u, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(jacobi_elliptic_sc)

} // namespace torchscience::cuda::special_functions
