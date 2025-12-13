#pragma once

#include <torchscience/csrc/impl/special_functions/sine_integral_sin.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t sine_integral_sin(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_sin(x);
}

template <typename scalar_t>
__device__ scalar_t sine_integral_sin_backward(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_sin_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(sine_integral_sin)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(sine_integral_sin)

} // namespace torchscience::cuda::special_functions
