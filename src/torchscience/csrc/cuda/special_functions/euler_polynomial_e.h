#pragma once

#include <torchscience/csrc/impl/special_functions/euler_polynomial_e.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t euler_polynomial_e(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::euler_polynomial_e(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> euler_polynomial_e_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::euler_polynomial_e_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(euler_polynomial_e, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(euler_polynomial_e)

} // namespace torchscience::cuda::special_functions
