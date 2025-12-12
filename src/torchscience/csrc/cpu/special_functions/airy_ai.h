#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/airy_ai.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T airy_ai(T x) {
  return torchscience::impl::special_functions::airy_ai(x);
}

template <typename T>
T airy_ai_backward(T x) {
  return torchscience::impl::special_functions::airy_ai_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(airy_ai)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(airy_ai)

} // namespace torchscience::cpu::special_functions
