#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_elliptic_integral_k.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T complete_elliptic_integral_k(T x) {
  return torchscience::impl::special_functions::complete_elliptic_integral_k(x);
}

template <typename T>
T complete_elliptic_integral_k_backward(T x) {
  return torchscience::impl::special_functions::complete_elliptic_integral_k_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(complete_elliptic_integral_k)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(complete_elliptic_integral_k)

} // namespace torchscience::cpu::special_functions
