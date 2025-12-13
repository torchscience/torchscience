#pragma once

#include <torchscience/csrc/impl/special_functions/prime_number_p.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t prime_number_p(scalar_t x) {
  return torchscience::impl::special_functions::prime_number_p(x);
}

template <typename scalar_t>
__device__ scalar_t prime_number_p_backward(scalar_t x) {
  return torchscience::impl::special_functions::prime_number_p_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(prime_number_p)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(prime_number_p)

} // namespace torchscience::cuda::special_functions
