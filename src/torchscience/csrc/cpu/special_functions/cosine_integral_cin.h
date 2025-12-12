#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/cosine_integral_cin.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T cosine_integral_cin(T x) {
  return torchscience::impl::special_functions::cosine_integral_cin(x);
}

template <typename T>
T cosine_integral_cin_backward(T x) {
  return torchscience::impl::special_functions::cosine_integral_cin_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(cosine_integral_cin)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(cosine_integral_cin)

} // namespace torchscience::cpu::special_functions
