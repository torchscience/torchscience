#pragma once

#include <torchscience/csrc/impl/special_functions/incomplete_legendre_elliptic_integral_d.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t incomplete_legendre_elliptic_integral_d(scalar_t phi, scalar_t k) {
  return torchscience::impl::special_functions::incomplete_legendre_elliptic_integral_d(phi, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> incomplete_legendre_elliptic_integral_d_backward(scalar_t phi, scalar_t k) {
  auto [grad_phi, grad_k] = torchscience::impl::special_functions::incomplete_legendre_elliptic_integral_d_backward(phi, k);
  return std::make_tuple(grad_phi, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::cuda::special_functions
