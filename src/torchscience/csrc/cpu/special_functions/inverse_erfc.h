#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/inverse_erfc.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T inverse_erfc(T x) {
  return torchscience::impl::special_functions::inverse_erfc(x);
}

template <typename T>
T inverse_erfc_backward(T x) {
  return torchscience::impl::special_functions::inverse_erfc_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(inverse_erfc)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(inverse_erfc)

} // namespace torchscience::cpu::special_functions
