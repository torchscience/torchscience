#pragma once

#include <torchscience/csrc/impl/special_functions/modified_bessel_k_derivative.h>
#include <torchscience/csrc/cuda/macros.h>

namespace torchscience::cuda::special_functions {

template <typename scalar_t>
__device__ scalar_t modified_bessel_k_derivative(scalar_t nu, scalar_t x) {
  return torchscience::impl::special_functions::modified_bessel_k_derivative(nu, x);
}

template <typename scalar_t>
__device__ std::tuple<scalar_t, scalar_t> modified_bessel_k_derivative_backward(scalar_t nu, scalar_t x) {
  auto [grad_nu, grad_x] = torchscience::impl::special_functions::modified_bessel_k_derivative_backward(nu, x);
  return std::make_tuple(grad_nu, grad_x);
}

TORCHSCIENCE_BINARY_CUDA_KERNEL(modified_bessel_k_derivative, nu, x)

TORCHSCIENCE_BINARY_CUDA_KERNEL_IMPL(modified_bessel_k_derivative)

} // namespace torchscience::cuda::special_functions
