#pragma once

#include <torchscience/csrc/impl/special_functions/complete_carlson_elliptic_r_g.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t complete_carlson_elliptic_r_g(scalar_t x, scalar_t y) {
  return torchscience::impl::special_functions::complete_carlson_elliptic_r_g(x, y);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> complete_carlson_elliptic_r_g_backward(scalar_t x, scalar_t y) {
  auto [grad_x, grad_y] = torchscience::impl::special_functions::complete_carlson_elliptic_r_g_backward(x, y);
  return std::make_tuple(grad_x, grad_y);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(complete_carlson_elliptic_r_g, x, y)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::cuda::special_functions
