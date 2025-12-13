#pragma once

#include <torchscience/csrc/impl/special_functions/bulirsch_elliptic_integral_el1.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t bulirsch_elliptic_integral_el1(scalar_t x, scalar_t kc) {
  return torchscience::impl::special_functions::bulirsch_elliptic_integral_el1(x, kc);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> bulirsch_elliptic_integral_el1_backward(scalar_t x, scalar_t kc) {
  auto [grad_x, grad_kc] = torchscience::impl::special_functions::bulirsch_elliptic_integral_el1_backward(x, kc);
  return std::make_tuple(grad_x, grad_kc);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(bulirsch_elliptic_integral_el1, x, kc)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(bulirsch_elliptic_integral_el1)

} // namespace torchscience::cuda::special_functions
