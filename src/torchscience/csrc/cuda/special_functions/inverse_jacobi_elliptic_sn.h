#pragma once

#include <torchscience/csrc/impl/special_functions/inverse_jacobi_elliptic_sn.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t inverse_jacobi_elliptic_sn(scalar_t x, scalar_t k) {
  return torchscience::impl::special_functions::inverse_jacobi_elliptic_sn(x, k);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> inverse_jacobi_elliptic_sn_backward(scalar_t x, scalar_t k) {
  auto [grad_x, grad_k] = torchscience::impl::special_functions::inverse_jacobi_elliptic_sn_backward(x, k);
  return std::make_tuple(grad_x, grad_k);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(inverse_jacobi_elliptic_sn, x, k)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(inverse_jacobi_elliptic_sn)

} // namespace torchscience::cuda::special_functions
