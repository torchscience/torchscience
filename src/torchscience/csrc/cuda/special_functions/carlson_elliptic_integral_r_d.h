#pragma once

#include <torchscience/csrc/impl/special_functions/carlson_elliptic_integral_r_d.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t carlson_elliptic_integral_r_d(scalar_t x, scalar_t y, scalar_t z) {
  return torchscience::impl::special_functions::carlson_elliptic_integral_r_d(x, y, z);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> carlson_elliptic_integral_r_d_backward(scalar_t x, scalar_t y, scalar_t z) {
  auto [grad_x, grad_y, grad_z] = torchscience::impl::special_functions::carlson_elliptic_integral_r_d_backward(x, y, z);
  return std::make_tuple(grad_x, grad_y, grad_z);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(carlson_elliptic_integral_r_d, x, y, z)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(carlson_elliptic_integral_r_d)

} // namespace torchscience::cuda::special_functions
