#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/sinhc_pi.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T sinhc_pi(T x) {
  return torchscience::impl::special_functions::sinhc_pi(x);
}

template <typename T>
c10::complex<T> sinhc_pi(c10::complex<T> z) {
  return torchscience::impl::special_functions::sinhc_pi(z);
}

template <typename T>
T sinhc_pi_backward(T x) {
  return torchscience::impl::special_functions::sinhc_pi_backward(x);
}

template <typename T>
c10::complex<T> sinhc_pi_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::sinhc_pi_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(sinhc_pi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(sinhc_pi)

} // namespace torchscience::cpu::special_functions
