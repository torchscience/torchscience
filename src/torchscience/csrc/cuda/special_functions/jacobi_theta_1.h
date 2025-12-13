#pragma once

#include <torchscience/csrc/impl/special_functions/jacobi_theta_1.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t jacobi_theta_1(scalar_t z, scalar_t q) {
  return torchscience::impl::special_functions::jacobi_theta_1(z, q);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> jacobi_theta_1_backward(scalar_t z, scalar_t q) {
  auto [grad_z, grad_q] = torchscience::impl::special_functions::jacobi_theta_1_backward(z, q);
  return std::make_tuple(grad_z, grad_q);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(jacobi_theta_1, z, q)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(jacobi_theta_1)

} // namespace torchscience::cuda::special_functions
