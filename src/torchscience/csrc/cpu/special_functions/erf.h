#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/erf.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T erf(T x) {
  return torchscience::impl::special_functions::erf(x);
}

template <typename T>
c10::complex<T> erf(c10::complex<T> z) {
  return torchscience::impl::special_functions::erf(z);
}

template <typename T>
T erf_backward(T x) {
  return torchscience::impl::special_functions::erf_backward(x);
}

template <typename T>
c10::complex<T> erf_backward(c10::complex<T> z) {
  return torchscience::impl::special_functions::erf_backward(z);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(erf)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(erf)

} // namespace torchscience::cpu::special_functions
