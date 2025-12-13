#pragma once

#include <torchscience/csrc/impl/special_functions/shifted_chebyshev_polynomial_u.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t shifted_chebyshev_polynomial_u(scalar_t n, scalar_t x) {
  return torchscience::impl::special_functions::shifted_chebyshev_polynomial_u(n, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> shifted_chebyshev_polynomial_u_backward(scalar_t n, scalar_t x) {
  auto [grad_n, grad_x] = torchscience::impl::special_functions::shifted_chebyshev_polynomial_u_backward(n, x);
  return std::make_tuple(grad_n, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(shifted_chebyshev_polynomial_u, n, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(shifted_chebyshev_polynomial_u)

} // namespace torchscience::cuda::special_functions
