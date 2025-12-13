#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/error_erfc.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T error_erfc(T x) {
  return torchscience::impl::special_functions::error_erfc(x);
}

template <typename T>
c10::complex<T> error_erfc(c10::complex<T> z) {
  return torchscience::impl::special_functions::error_erfc(z);
}

template <typename T>
T error_erfc_backward(T x) {
  return torchscience::impl::special_functions::error_erfc_backward(x);
}

template <typename T>
c10::complex<T> error_erfc_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::error_erfc_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(error_erfc)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(error_erfc)

} // namespace torchscience::cpu::special_functions
