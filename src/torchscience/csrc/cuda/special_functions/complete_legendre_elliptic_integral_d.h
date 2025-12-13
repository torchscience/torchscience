#pragma once

#include <torchscience/csrc/impl/special_functions/complete_legendre_elliptic_integral_d.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t complete_legendre_elliptic_integral_d(scalar_t x) {
  return torchscience::impl::special_functions::complete_legendre_elliptic_integral_d(x);
}

template <typename scalar_t>
__device__ scalar_t complete_legendre_elliptic_integral_d_backward(scalar_t x) {
  return torchscience::impl::special_functions::complete_legendre_elliptic_integral_d_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::cuda::special_functions
