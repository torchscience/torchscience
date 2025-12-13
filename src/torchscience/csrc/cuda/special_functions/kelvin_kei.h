#pragma once

#include <torchscience/csrc/impl/special_functions/kelvin_kei.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t kelvin_kei(scalar_t v, scalar_t x) {
  return torchscience::impl::special_functions::kelvin_kei(v, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> kelvin_kei_backward(scalar_t v, scalar_t x) {
  auto [grad_v, grad_x] = torchscience::impl::special_functions::kelvin_kei_backward(v, x);
  return std::make_tuple(grad_v, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(kelvin_kei, v, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(kelvin_kei)

} // namespace torchscience::cuda::special_functions
