#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/erfc.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T erfc(T x) {
  return torchscience::impl::special_functions::erfc(x);
}

template <typename T>
c10::complex<T> erfc(c10::complex<T> z) {
  return torchscience::impl::special_functions::erfc(z);
}

template <typename T>
T erfc_backward(T x) {
  return torchscience::impl::special_functions::erfc_backward(x);
}

template <typename T>
c10::complex<T> erfc_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::erfc_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(erfc)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(erfc)

} // namespace torchscience::cpu::special_functions
