#pragma once

#include <torchscience/csrc/impl/special_functions/carlson_elliptic_integral_r_j.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t carlson_elliptic_integral_r_j(scalar_t x, scalar_t y, scalar_t z, scalar_t p) {
  return torchscience::impl::special_functions::carlson_elliptic_integral_r_j(x, y, z, p);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> carlson_elliptic_integral_r_j_backward(scalar_t x, scalar_t y, scalar_t z, scalar_t p) {
  auto [grad_x, grad_y, grad_z, grad_p] = torchscience::impl::special_functions::carlson_elliptic_integral_r_j_backward(x, y, z, p);
  return std::make_tuple(grad_x, grad_y, grad_z, grad_p);
}

TORCHSCIENCE_QUATERNARY_CUDA_KERNEL(carlson_elliptic_integral_r_j, x, y, z, p)

TORCHSCIENCE_QUATERNARY_CUDA_KERNEL_IMPL(carlson_elliptic_integral_r_j)

} // namespace torchscience::cuda::special_functions
