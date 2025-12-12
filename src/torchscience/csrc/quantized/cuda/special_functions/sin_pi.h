#pragma once

#include <torchscience/csrc/impl/special_functions/sin_pi.h>
#include <torchscience/csrc/quantized/cuda/macros.h>

namespace torchscience::quantized::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t sin_pi(scalar_t x) {
  return torchscience::impl::special_functions::sin_pi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CUDA_KERNEL(sin_pi)

TORCHSCIENCE_UNARY_QUANTIZED_CUDA_KERNEL_IMPL(sin_pi)

} // namespace torchscience::quantized::cuda::special_functions
