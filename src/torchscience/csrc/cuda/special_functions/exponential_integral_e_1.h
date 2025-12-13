#pragma once

#include <torchscience/csrc/impl/special_functions/exponential_integral_e_1.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t exponential_integral_e_1(scalar_t x) {
  return torchscience::impl::special_functions::exponential_integral_e_1(x);
}

template <typename scalar_t>
__device__ scalar_t exponential_integral_e_1_backward(scalar_t x) {
  return torchscience::impl::special_functions::exponential_integral_e_1_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(exponential_integral_e_1)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(exponential_integral_e_1)

} // namespace torchscience::cuda::special_functions
