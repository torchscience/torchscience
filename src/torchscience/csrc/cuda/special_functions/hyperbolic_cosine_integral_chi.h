#pragma once

#include <torchscience/csrc/impl/special_functions/hyperbolic_cosine_integral_chi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t hyperbolic_cosine_integral_chi(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_cosine_integral_chi(x);
}

template <typename scalar_t>
__device__ scalar_t hyperbolic_cosine_integral_chi_backward(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_cosine_integral_chi_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::cuda::special_functions
