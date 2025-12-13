#pragma once

#include <torchscience/csrc/impl/special_functions/hankel_h_1.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t hankel_h_1(scalar_t nu, scalar_t x) {
  return torchscience::impl::special_functions::hankel_h_1(nu, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> hankel_h_1_backward(scalar_t nu, scalar_t x) {
  auto [grad_nu, grad_x] = torchscience::impl::special_functions::hankel_h_1_backward(nu, x);
  return std::make_tuple(grad_nu, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(hankel_h_1, nu, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(hankel_h_1)

} // namespace torchscience::cuda::special_functions
