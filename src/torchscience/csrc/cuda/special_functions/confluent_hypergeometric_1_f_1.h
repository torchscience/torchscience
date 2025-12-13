#pragma once

#include <torchscience/csrc/impl/special_functions/confluent_hypergeometric_1_f_1.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t confluent_hypergeometric_1_f_1(scalar_t a, scalar_t b, scalar_t z) {
  return torchscience::impl::special_functions::confluent_hypergeometric_1_f_1(a, b, z);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> confluent_hypergeometric_1_f_1_backward(scalar_t a, scalar_t b, scalar_t z) {
  auto [grad_a, grad_b, grad_z] = torchscience::impl::special_functions::confluent_hypergeometric_1_f_1_backward(a, b, z);
  return std::make_tuple(grad_a, grad_b, grad_z);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(confluent_hypergeometric_1_f_1, a, b, z)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::cuda::special_functions
