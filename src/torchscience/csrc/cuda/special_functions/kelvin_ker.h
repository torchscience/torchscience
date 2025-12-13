#pragma once

#include <torchscience/csrc/impl/special_functions/kelvin_ker.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t kelvin_ker(scalar_t v, scalar_t x) {
  return torchscience::impl::special_functions::kelvin_ker(v, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> kelvin_ker_backward(scalar_t v, scalar_t x) {
  auto [grad_v, grad_x] = torchscience::impl::special_functions::kelvin_ker_backward(v, x);
  return std::make_tuple(grad_v, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(kelvin_ker, v, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(kelvin_ker)

} // namespace torchscience::cuda::special_functions
