#pragma once

#include <torchscience/csrc/impl/special_functions/rising_factorial.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t rising_factorial(scalar_t x, scalar_t n) {
  return torchscience::impl::special_functions::rising_factorial(x, n);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> rising_factorial_backward(scalar_t x, scalar_t n) {
  auto [grad_x, grad_n] = torchscience::impl::special_functions::rising_factorial_backward(x, n);
  return std::make_tuple(grad_x, grad_n);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(rising_factorial, x, n)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(rising_factorial)

} // namespace torchscience::cuda::special_functions
