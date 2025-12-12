#pragma once

#include <torchscience/csrc/impl/special_functions/sin_pi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t sin_pi(scalar_t x) {
  return torchscience::impl::special_functions::sin_pi(x);
}

template <typename scalar_t>
__device__ scalar_t sin_pi_backward(scalar_t x) {
  return torchscience::impl::special_functions::sin_pi_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(sin_pi)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(sin_pi)

} // namespace torchscience::cuda::special_functions
