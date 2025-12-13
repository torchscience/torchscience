#pragma once

#include <torchscience/csrc/impl/special_functions/error_erfc.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t error_erfc(scalar_t x) {
  return torchscience::impl::special_functions::error_erfc(x);
}

template <typename scalar_t>
__device__ scalar_t error_erfc_backward(scalar_t x) {
  return torchscience::impl::special_functions::error_erfc_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(error_erfc)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(error_erfc)

} // namespace torchscience::cuda::special_functions
