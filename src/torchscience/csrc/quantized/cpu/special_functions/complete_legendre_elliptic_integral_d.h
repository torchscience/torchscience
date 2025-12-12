#pragma once

#include <torchscience/csrc/impl/special_functions/complete_legendre_elliptic_integral_d.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t complete_legendre_elliptic_integral_d(scalar_t x) {
  return torchscience::impl::special_functions::complete_legendre_elliptic_integral_d(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::quantized::cpu::special_functions
