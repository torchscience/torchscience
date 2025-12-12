#pragma once

#include <torchscience/csrc/impl/special_functions/bessel_j_derivative.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t bessel_j_derivative(scalar_t nu, scalar_t x) {
  return torchscience::impl::special_functions::bessel_j_derivative(nu, x);
}

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(bessel_j_derivative, nu, x)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(bessel_j_derivative)

} // namespace torchscience::quantized::cpu::special_functions
