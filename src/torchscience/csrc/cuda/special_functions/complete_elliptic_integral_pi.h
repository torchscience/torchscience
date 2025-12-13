#pragma once

#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_pi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t complete_elliptic_integral_pi(scalar_t n, scalar_t k) {
  return torchscience::impl::special_functions::complete_elliptic_integral_pi(n, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> complete_elliptic_integral_pi_backward(scalar_t n, scalar_t k) {
  auto [grad_n, grad_k] = torchscience::impl::special_functions::complete_elliptic_integral_pi_backward(n, k);
  return std::make_tuple(grad_n, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(complete_elliptic_integral_pi, n, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(complete_elliptic_integral_pi)

} // namespace torchscience::cuda::special_functions
