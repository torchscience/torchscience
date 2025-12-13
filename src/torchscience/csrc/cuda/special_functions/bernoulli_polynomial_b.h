#pragma once

#include <torchscience/csrc/impl/special_functions/bernoulli_polynomial_b.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t bernoulli_polynomial_b(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::bernoulli_polynomial_b(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> bernoulli_polynomial_b_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::bernoulli_polynomial_b_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(bernoulli_polynomial_b, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(bernoulli_polynomial_b)

} // namespace torchscience::cuda::special_functions
