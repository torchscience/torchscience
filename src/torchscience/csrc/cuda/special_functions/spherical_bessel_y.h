#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_bessel_y.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t spherical_bessel_y(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::spherical_bessel_y(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> spherical_bessel_y_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::spherical_bessel_y_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(spherical_bessel_y, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(spherical_bessel_y)

} // namespace torchscience::cuda::special_functions
