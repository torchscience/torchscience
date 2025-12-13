#pragma once

#include <torchscience/csrc/impl/special_functions/tangent_number_t_2.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t tangent_number_t_2(scalar_t x) {
  return torchscience::impl::special_functions::tangent_number_t_2(x);
}

template <typename scalar_t>
__device__ scalar_t tangent_number_t_2_backward(scalar_t x) {
  return torchscience::impl::special_functions::tangent_number_t_2_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(tangent_number_t_2)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(tangent_number_t_2)

} // namespace torchscience::cuda::special_functions
