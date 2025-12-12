#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/cosine_integral_ci.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T cosine_integral_ci(T x) {
  return torchscience::impl::special_functions::cosine_integral_ci(x);
}

template <typename T>
T cosine_integral_ci_backward(T x) {
  return torchscience::impl::special_functions::cosine_integral_ci_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(cosine_integral_ci)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(cosine_integral_ci)

} // namespace torchscience::cpu::special_functions
