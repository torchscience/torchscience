#pragma once

#include <torchscience/csrc/impl/special_functions/euler_number_e.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t euler_number_e(scalar_t x) {
  return torchscience::impl::special_functions::euler_number_e(x);
}

template <typename scalar_t>
__device__ scalar_t euler_number_e_backward(scalar_t x) {
  return torchscience::impl::special_functions::euler_number_e_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(euler_number_e)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(euler_number_e)

} // namespace torchscience::cuda::special_functions
