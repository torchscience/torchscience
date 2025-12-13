#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/error_inverse_erfc.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T error_inverse_erfc(T x) {
  return torchscience::impl::special_functions::error_inverse_erfc(x);
}

template <typename T>
T error_inverse_erfc_backward(T x) {
  return torchscience::impl::special_functions::error_inverse_erfc_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(error_inverse_erfc)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(error_inverse_erfc)

} // namespace torchscience::cpu::special_functions
