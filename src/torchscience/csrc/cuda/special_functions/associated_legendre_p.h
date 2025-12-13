#pragma once

#include <torchscience/csrc/impl/special_functions/associated_legendre_p.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t associated_legendre_p(scalar_t n, scalar_t m, scalar_t x) {
  return torchscience::impl::special_functions::associated_legendre_p(n, m, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t, scalar_t> associated_legendre_p_backward(scalar_t n, scalar_t m, scalar_t x) {
  auto [grad_n, grad_m, grad_x] = torchscience::impl::special_functions::associated_legendre_p_backward(n, m, x);
  return std::make_tuple(grad_n, grad_m, grad_x);
}

TORCHSCIENCE_TERNARY_CUDA_KERNEL(associated_legendre_p, n, m, x)

TORCHSCIENCE_TERNARY_CUDA_KERNEL_IMPL(associated_legendre_p)

} // namespace torchscience::cuda::special_functions
