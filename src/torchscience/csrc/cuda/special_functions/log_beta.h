#pragma once

#include <torchscience/csrc/impl/special_functions/log_beta.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t log_beta(scalar_t a, scalar_t b) {
  return torchscience::impl::special_functions::log_beta(a, b);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> log_beta_backward(scalar_t a, scalar_t b) {
  auto [grad_a, grad_b] = torchscience::impl::special_functions::log_beta_backward(a, b);
  return std::make_tuple(grad_a, grad_b);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(log_beta, a, b)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(log_beta)

} // namespace torchscience::cuda::special_functions
