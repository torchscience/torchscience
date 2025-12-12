#pragma once

#include <torchscience/csrc/impl/special_functions/exponential_integral_e_1.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t exponential_integral_e_1(scalar_t x) {
  return torchscience::impl::special_functions::exponential_integral_e_1(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(exponential_integral_e_1)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(exponential_integral_e_1)

} // namespace torchscience::quantized::cpu::special_functions
