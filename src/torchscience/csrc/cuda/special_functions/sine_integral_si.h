#pragma once

#include <torchscience/csrc/impl/special_functions/sine_integral_si.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t sine_integral_si(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_si(x);
}

template <typename scalar_t>
__device__ scalar_t sine_integral_si_backward(scalar_t x) {
  return torchscience::impl::special_functions::sine_integral_si_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(sine_integral_si)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(sine_integral_si)

} // namespace torchscience::cuda::special_functions
