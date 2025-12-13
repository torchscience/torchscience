#pragma once

#include <torchscience/csrc/impl/special_functions/bessel_y.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t bessel_y(scalar_t nu, scalar_t x) {
  return torchscience::impl::special_functions::bessel_y(nu, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> bessel_y_backward(scalar_t nu, scalar_t x) {
  auto [grad_nu, grad_x] = torchscience::impl::special_functions::bessel_y_backward(nu, x);
  return std::make_tuple(grad_nu, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(bessel_y, nu, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(bessel_y)

} // namespace torchscience::cuda::special_functions
