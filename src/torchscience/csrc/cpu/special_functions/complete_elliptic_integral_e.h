#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_e.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T complete_elliptic_integral_e(T x) {
  return torchscience::impl::special_functions::complete_elliptic_integral_e(x);
}

template <typename T>
T complete_elliptic_integral_e_backward(T x) {
  return torchscience::impl::special_functions::complete_elliptic_integral_e_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(complete_elliptic_integral_e)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(complete_elliptic_integral_e)

} // namespace torchscience::cpu::special_functions
