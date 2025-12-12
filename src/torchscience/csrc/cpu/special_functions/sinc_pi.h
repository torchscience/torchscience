#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/sinc_pi.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T sinc_pi(T x) {
  return torchscience::impl::special_functions::sinc_pi(x);
}

template <typename T>
c10::complex<T> sinc_pi(c10::complex<T> z) {
  return torchscience::impl::special_functions::sinc_pi(z);
}

template <typename T>
T sinc_pi_backward(T x) {
  return torchscience::impl::special_functions::sinc_pi_backward(x);
}

template <typename T>
c10::complex<T> sinc_pi_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::sinc_pi_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(sinc_pi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(sinc_pi)

} // namespace torchscience::cpu::special_functions
