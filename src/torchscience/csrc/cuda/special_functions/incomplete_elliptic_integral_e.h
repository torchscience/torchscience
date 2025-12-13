#pragma once

#include <torchscience/csrc/impl/special_functions/incomplete_elliptic_integral_e.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t incomplete_elliptic_integral_e(scalar_t phi, scalar_t k) {
  return torchscience::impl::special_functions::incomplete_elliptic_integral_e(phi, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> incomplete_elliptic_integral_e_backward(scalar_t phi, scalar_t k) {
  auto [grad_phi, grad_k] = torchscience::impl::special_functions::incomplete_elliptic_integral_e_backward(phi, k);
  return std::make_tuple(grad_phi, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(incomplete_elliptic_integral_e, phi, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(incomplete_elliptic_integral_e)

} // namespace torchscience::cuda::special_functions
