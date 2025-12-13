#pragma once

#include <torchscience/csrc/impl/special_functions/bernoulli_number_b.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t bernoulli_number_b(scalar_t x) {
  return torchscience::impl::special_functions::bernoulli_number_b(x);
}

template <typename scalar_t>
__device__ scalar_t bernoulli_number_b_backward(scalar_t x) {
  return torchscience::impl::special_functions::bernoulli_number_b_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(bernoulli_number_b)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(bernoulli_number_b)

} // namespace torchscience::cuda::special_functions
