#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/complete_legendre_elliptic_integral_d.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T complete_legendre_elliptic_integral_d(T x) {
  return torchscience::impl::special_functions::complete_legendre_elliptic_integral_d(x);
}

template <typename T>
T complete_legendre_elliptic_integral_d_backward(T x) {
  return torchscience::impl::special_functions::complete_legendre_elliptic_integral_d_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::cpu::special_functions
