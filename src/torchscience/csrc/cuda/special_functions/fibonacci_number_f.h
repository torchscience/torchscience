#pragma once

#include <torchscience/csrc/impl/special_functions/fibonacci_number_f.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t fibonacci_number_f(scalar_t x) {
  return torchscience::impl::special_functions::fibonacci_number_f(x);
}

template <typename scalar_t>
__device__ scalar_t fibonacci_number_f_backward(scalar_t x) {
  return torchscience::impl::special_functions::fibonacci_number_f_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(fibonacci_number_f)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(fibonacci_number_f)

} // namespace torchscience::cuda::special_functions
