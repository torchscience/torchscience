#pragma once

#include <torchscience/csrc/impl/special_functions/jacobi_theta_4.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t jacobi_theta_4(scalar_t z, scalar_t q) {
  return torchscience::impl::special_functions::jacobi_theta_4(z, q);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> jacobi_theta_4_backward(scalar_t z, scalar_t q) {
  auto [grad_z, grad_q] = torchscience::impl::special_functions::jacobi_theta_4_backward(z, q);
  return std::make_tuple(grad_z, grad_q);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(jacobi_theta_4, z, q)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(jacobi_theta_4)

} // namespace torchscience::cuda::special_functions
