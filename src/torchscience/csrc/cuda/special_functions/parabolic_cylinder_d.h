#pragma once

#include <torchscience/csrc/impl/special_functions/parabolic_cylinder_d.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t parabolic_cylinder_d(scalar_t nu, scalar_t z) {
  return torchscience::impl::special_functions::parabolic_cylinder_d(nu, z);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> parabolic_cylinder_d_backward(scalar_t nu, scalar_t z) {
  auto [grad_nu, grad_z] = torchscience::impl::special_functions::parabolic_cylinder_d_backward(nu, z);
  return std::make_tuple(grad_nu, grad_z);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(parabolic_cylinder_d, nu, z)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(parabolic_cylinder_d)

} // namespace torchscience::cuda::special_functions
