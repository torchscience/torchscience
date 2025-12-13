#pragma once

#include <torchscience/csrc/impl/special_functions/legendre_elliptic_integral_pi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t legendre_elliptic_integral_pi(scalar_t n, scalar_t phi, scalar_t k) {
  return torchscience::impl::special_functions::legendre_elliptic_integral_pi(n, phi, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> legendre_elliptic_integral_pi_backward(scalar_t n, scalar_t phi, scalar_t k) {
  auto [grad_n, grad_phi, grad_k] = torchscience::impl::special_functions::legendre_elliptic_integral_pi_backward(n, phi, k);
  return std::make_tuple(grad_n, grad_phi, grad_k);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(legendre_elliptic_integral_pi, n, phi, k)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(legendre_elliptic_integral_pi)

} // namespace torchscience::cuda::special_functions
