#pragma once

#include <torchscience/csrc/impl/special_functions/cosine_integral_cin.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t cosine_integral_cin(scalar_t x) {
  return torchscience::impl::special_functions::cosine_integral_cin(x);
}

template <typename scalar_t>
__device__ scalar_t cosine_integral_cin_backward(scalar_t x) {
  return torchscience::impl::special_functions::cosine_integral_cin_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(cosine_integral_cin)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(cosine_integral_cin)

} // namespace torchscience::cuda::special_functions
