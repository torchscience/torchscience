#pragma once

#include <torchscience/csrc/impl/special_functions/sin_pi.h>
#include <torchscience/csrc/cpu/macros.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T sin_pi(T x) {
  return torchscience::impl::special_functions::sin_pi(x);
}

template <typename T>
c10::complex<T> sin_pi(c10::complex<T> z) {
  return torchscience::impl::special_functions::sin_pi(z);
}

template <typename T>
T sin_pi_backward(T x) {
  return torchscience::impl::special_functions::sin_pi_backward(x);
}

template <typename T>
c10::complex<T> sin_pi_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::sin_pi_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(sin_pi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(sin_pi)

} // namespace torchscience::cpu::special_functions
