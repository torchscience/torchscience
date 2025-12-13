#pragma once

#include <torchscience/csrc/impl/special_functions/error_inverse_erf.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t error_inverse_erf(scalar_t x) {
  return torchscience::impl::special_functions::error_inverse_erf(x);
}

template <typename scalar_t>
__device__ scalar_t error_inverse_erf_backward(scalar_t x) {
  return torchscience::impl::special_functions::error_inverse_erf_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(error_inverse_erf)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(error_inverse_erf)

} // namespace torchscience::cuda::special_functions
