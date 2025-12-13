#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_modified_bessel_i.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t spherical_modified_bessel_i(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::spherical_modified_bessel_i(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> spherical_modified_bessel_i_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::spherical_modified_bessel_i_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(spherical_modified_bessel_i, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(spherical_modified_bessel_i)

} // namespace torchscience::cuda::special_functions
