#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/gamma.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T gamma(T x) {
  return torchscience::impl::special_functions::gamma(x);
}

template <typename T>
c10::complex<T> gamma(c10::complex<T> z) {
  return torchscience::impl::special_functions::gamma(z);
}

template <typename T>
T gamma_backward(T x) {
  return torchscience::impl::special_functions::gamma_backward(x);
}

template <typename T>
c10::complex<T> gamma_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::gamma_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(gamma)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(gamma)

} // namespace torchscience::cpu::special_functions
