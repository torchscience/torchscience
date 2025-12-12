#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/logarithmic_integral_li.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T logarithmic_integral_li(T x) {
  return torchscience::impl::special_functions::logarithmic_integral_li(x);
}

template <typename T>
T logarithmic_integral_li_backward(T x) {
  return torchscience::impl::special_functions::logarithmic_integral_li_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(logarithmic_integral_li)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(logarithmic_integral_li)

} // namespace torchscience::cpu::special_functions
