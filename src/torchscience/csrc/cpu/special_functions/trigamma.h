#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/trigamma.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T trigamma(T x) {
  return torchscience::impl::special_functions::trigamma(x);
}

template <typename T>
c10::complex<T> trigamma(c10::complex<T> z) {
  return torchscience::impl::special_functions::trigamma(z);
}

template <typename T>
T trigamma_backward(T x) {
  return torchscience::impl::special_functions::trigamma_backward(x);
}

template <typename T>
c10::complex<T> trigamma_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::trigamma_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(trigamma)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(trigamma)

} // namespace torchscience::cpu::special_functions
