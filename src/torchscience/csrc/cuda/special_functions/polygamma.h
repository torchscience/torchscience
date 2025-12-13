#pragma once

#include <torchscience/csrc/impl/special_functions/polygamma.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t polygamma(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::polygamma(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> polygamma_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::polygamma_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(polygamma, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(polygamma)

} // namespace torchscience::cuda::special_functions
