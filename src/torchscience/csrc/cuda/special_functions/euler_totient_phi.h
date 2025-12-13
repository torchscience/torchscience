#pragma once

#include <torchscience/csrc/impl/special_functions/euler_totient_phi.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t euler_totient_phi(scalar_t x) {
  return torchscience::impl::special_functions::euler_totient_phi(x);
}

template <typename scalar_t>
__device__ scalar_t euler_totient_phi_backward(scalar_t x) {
  return torchscience::impl::special_functions::euler_totient_phi_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(euler_totient_phi)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(euler_totient_phi)

} // namespace torchscience::cuda::special_functions
