#pragma once

#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_e.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t complete_elliptic_integral_e(scalar_t k) {
  return torchscience::impl::special_functions::complete_elliptic_integral_e(k);
}

template <typename scalar_t>
__device__ scalar_t complete_elliptic_integral_e_backward(scalar_t k) {
  return torchscience::impl::special_functions::complete_elliptic_integral_e_backward(k);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(complete_elliptic_integral_e)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(complete_elliptic_integral_e)

} // namespace torchscience::cuda::special_functions
