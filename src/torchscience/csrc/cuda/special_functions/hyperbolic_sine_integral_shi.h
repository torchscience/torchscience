#pragma once

#include <torchscience/csrc/impl/special_functions/hyperbolic_sine_integral_shi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t hyperbolic_sine_integral_shi(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_sine_integral_shi(x);
}

template <typename scalar_t>
__device__ scalar_t hyperbolic_sine_integral_shi_backward(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_sine_integral_shi_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::cuda::special_functions
