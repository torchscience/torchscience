#pragma once

#include <torchscience/csrc/impl/special_functions/cosine_integral_ci.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t cosine_integral_ci(scalar_t x) {
  return torchscience::impl::special_functions::cosine_integral_ci(x);
}

template <typename scalar_t>
__device__ scalar_t cosine_integral_ci_backward(scalar_t x) {
  return torchscience::impl::special_functions::cosine_integral_ci_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(cosine_integral_ci)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(cosine_integral_ci)

} // namespace torchscience::cuda::special_functions
