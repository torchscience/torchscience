#pragma once

#include <torchscience/csrc/impl/special_functions/spherical_hankel_h_2.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t spherical_hankel_h_2(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::spherical_hankel_h_2(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> spherical_hankel_h_2_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::spherical_hankel_h_2_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(spherical_hankel_h_2, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(spherical_hankel_h_2)

} // namespace torchscience::cuda::special_functions
