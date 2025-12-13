#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/error_erf.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T error_erf(T x) {
  return torchscience::impl::special_functions::error_erf(x);
}

template <typename T>
c10::complex<T> error_erf(c10::complex<T> z) {
  return torchscience::impl::special_functions::error_erf(z);
}

template <typename T>
T error_erf_backward(T x) {
  return torchscience::impl::special_functions::error_erf_backward(x);
}

template <typename T>
c10::complex<T> error_erf_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::error_erf_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(error_erf)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(error_erf)

} // namespace torchscience::cpu::special_functions
