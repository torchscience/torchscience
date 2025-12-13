#pragma once

#include <torchscience/csrc/impl/special_functions/legendre_q.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t legendre_q(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::legendre_q(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> legendre_q_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::legendre_q_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(legendre_q, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(legendre_q)

} // namespace torchscience::cuda::special_functions
