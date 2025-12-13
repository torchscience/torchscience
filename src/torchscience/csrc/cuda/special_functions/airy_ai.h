#pragma once

#include <torchscience/csrc/impl/special_functions/airy_ai.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t airy_ai(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai(x);
}

template <typename scalar_t>
__device__ scalar_t airy_ai_backward(scalar_t x) {
  return torchscience::impl::special_functions::airy_ai_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(airy_ai)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(airy_ai)

} // namespace torchscience::cuda::special_functions
