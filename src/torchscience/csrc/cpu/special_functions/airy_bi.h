#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/airy_bi.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T airy_bi(T x) {
  return torchscience::impl::special_functions::airy_bi(x);
}

template <typename T>
T airy_bi_backward(T x) {
  return torchscience::impl::special_functions::airy_bi_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(airy_bi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(airy_bi)

} // namespace torchscience::cpu::special_functions
