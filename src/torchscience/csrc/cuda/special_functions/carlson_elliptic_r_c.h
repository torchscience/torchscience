#pragma once

#include <torchscience/csrc/impl/special_functions/carlson_elliptic_r_c.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t carlson_elliptic_r_c(scalar_t x, scalar_t y) {
  return torchscience::impl::special_functions::carlson_elliptic_r_c(x, y);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> carlson_elliptic_r_c_backward(scalar_t x, scalar_t y) {
  auto [grad_x, grad_y] = torchscience::impl::special_functions::carlson_elliptic_r_c_backward(x, y);
  return std::make_tuple(grad_x, grad_y);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(carlson_elliptic_r_c, x, y)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::cuda::special_functions
