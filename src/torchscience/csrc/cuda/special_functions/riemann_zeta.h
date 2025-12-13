#pragma once

#include <torchscience/csrc/impl/special_functions/riemann_zeta.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t riemann_zeta(scalar_t x) {
  return torchscience::impl::special_functions::riemann_zeta(x);
}

template <typename scalar_t>
__device__ scalar_t riemann_zeta_backward(scalar_t x) {
  return torchscience::impl::special_functions::riemann_zeta_backward(x);
}

TORCHSCIENCE_UNARY_CUDA_KERNEL(riemann_zeta)

TORCHSCIENCE_UNARY_CUDA_KERNEL_IMPL(riemann_zeta)

} // namespace torchscience::cuda::special_functions
