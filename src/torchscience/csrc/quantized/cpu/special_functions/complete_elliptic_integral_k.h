#pragma once

#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_k.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t complete_elliptic_integral_k(scalar_t x) {
  return torchscience::impl::special_functions::complete_elliptic_integral_k(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(complete_elliptic_integral_k)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(complete_elliptic_integral_k)

} // namespace torchscience::quantized::cpu::special_functions
