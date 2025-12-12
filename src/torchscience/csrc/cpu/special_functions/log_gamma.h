#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/log_gamma.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T log_gamma(T x) {
  return torchscience::impl::special_functions::log_gamma(x);
}

template <typename T>
c10::complex<T> log_gamma(c10::complex<T> z) {
  return torchscience::impl::special_functions::log_gamma(z);
}

template <typename T>
T log_gamma_backward(T x) {
  return torchscience::impl::special_functions::log_gamma_backward(x);
}

template <typename T>
c10::complex<T> log_gamma_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::log_gamma_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(log_gamma)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(log_gamma)

} // namespace torchscience::cpu::special_functions
