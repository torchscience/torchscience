#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/inverse_erf.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T inverse_erf(T x) {
  return torchscience::impl::special_functions::inverse_erf(x);
}

template <typename T>
T inverse_erf_backward(T x) {
  return torchscience::impl::special_functions::inverse_erf_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(inverse_erf)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(inverse_erf)

} // namespace torchscience::cpu::special_functions
