#pragma once

#include <torchscience/csrc/impl/special_functions/logarithmic_integral_li.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t logarithmic_integral_li(scalar_t x) {
  return torchscience::impl::special_functions::logarithmic_integral_li(x);
}

template <typename scalar_t>
__device__ scalar_t logarithmic_integral_li_backward(scalar_t x) {
  return torchscience::impl::special_functions::logarithmic_integral_li_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(logarithmic_integral_li)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(logarithmic_integral_li)

} // namespace torchscience::cuda::special_functions
