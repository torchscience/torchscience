#pragma once

#include <torchscience/csrc/impl/special_functions/exponential_integral_ei.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t exponential_integral_ei(scalar_t x) {
  return torchscience::impl::special_functions::exponential_integral_ei(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(exponential_integral_ei)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(exponential_integral_ei)

} // namespace torchscience::quantized::cpu::special_functions
