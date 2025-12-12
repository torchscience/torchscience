#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/digamma.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T digamma(T x) {
  return torchscience::impl::special_functions::digamma(x);
}

template <typename T>
c10::complex<T> digamma(c10::complex<T> z) {
  return torchscience::impl::special_functions::digamma(z);
}

template <typename T>
T digamma_backward(T x) {
  return torchscience::impl::special_functions::digamma_backward(x);
}

template <typename T>
c10::complex<T> digamma_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::digamma_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(digamma)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(digamma)

} // namespace torchscience::cpu::special_functions
