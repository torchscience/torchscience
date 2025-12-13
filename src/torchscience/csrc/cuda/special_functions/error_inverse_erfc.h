#pragma once

#include <torchscience/csrc/impl/special_functions/error_inverse_erfc.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t error_inverse_erfc(scalar_t x) {
  return torchscience::impl::special_functions::error_inverse_erfc(x);
}

template <typename scalar_t>
__device__ scalar_t error_inverse_erfc_backward(scalar_t x) {
  return torchscience::impl::special_functions::error_inverse_erfc_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(error_inverse_erfc)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(error_inverse_erfc)

} // namespace torchscience::cuda::special_functions
