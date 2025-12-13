#pragma once

#include <torchscience/csrc/impl/special_functions/binomial_coefficient.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t binomial_coefficient(scalar_t n, scalar_t k) {
  return torchscience::impl::special_functions::binomial_coefficient(n, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> binomial_coefficient_backward(scalar_t n, scalar_t k) {
  auto [grad_n, grad_k] = torchscience::impl::special_functions::binomial_coefficient_backward(n, k);
  return std::make_tuple(grad_n, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::cuda::special_functions
