#pragma once

#include <torchscience/csrc/impl/special_functions/modified_bessel_i_derivative.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t modified_bessel_i_derivative(scalar_t nu, scalar_t x) {
  return torchscience::impl::special_functions::modified_bessel_i_derivative(nu, x);
}

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL(modified_bessel_i_derivative, nu, x)

TORCHSCIENCE_BINARY_QUANTIZED_CPU_KERNEL_IMPL(modified_bessel_i_derivative)

} // namespace torchscience::quantized::cpu::special_functions
