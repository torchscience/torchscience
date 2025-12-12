#pragma once

#include <torchscience/csrc/impl/special_functions/logarithmic_integral_li.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t logarithmic_integral_li(scalar_t x) {
  return torchscience::impl::special_functions::logarithmic_integral_li(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(logarithmic_integral_li)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(logarithmic_integral_li)

} // namespace torchscience::quantized::cpu::special_functions
