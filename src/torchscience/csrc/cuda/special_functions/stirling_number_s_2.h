#pragma once

#include <torchscience/csrc/impl/special_functions/stirling_number_s_2.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t stirling_number_s_2(scalar_t n, scalar_t k) {
  return torchscience::impl::special_functions::stirling_number_s_2(n, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> stirling_number_s_2_backward(scalar_t n, scalar_t k) {
  auto [grad_n, grad_k] = torchscience::impl::special_functions::stirling_number_s_2_backward(n, k);
  return std::make_tuple(grad_n, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(stirling_number_s_2, n, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(stirling_number_s_2)

} // namespace torchscience::cuda::special_functions
