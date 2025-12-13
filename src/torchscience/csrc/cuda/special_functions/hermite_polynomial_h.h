#pragma once

#include <torchscience/csrc/impl/special_functions/hermite_polynomial_h.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t hermite_polynomial_h(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::hermite_polynomial_h(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> hermite_polynomial_h_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::hermite_polynomial_h_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(hermite_polynomial_h, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(hermite_polynomial_h)

} // namespace torchscience::cuda::special_functions
