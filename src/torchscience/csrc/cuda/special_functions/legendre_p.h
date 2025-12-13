#pragma once

#include <torchscience/csrc/impl/special_functions/legendre_p.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t legendre_p(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::legendre_p(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> legendre_p_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::legendre_p_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(legendre_p, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(legendre_p)

} // namespace torchscience::cuda::special_functions
