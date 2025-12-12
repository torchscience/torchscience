#pragma once

#include <torchscience/csrc/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/hyperbolic_cosine_integral_chi.h>

namespace torchscience::cpu::special_functions {

template <typename T>
T hyperbolic_cosine_integral_chi(T x) {
  return torchscience::impl::special_functions::hyperbolic_cosine_integral_chi(x);
}

template <typename T>
T hyperbolic_cosine_integral_chi_backward(T x) {
  return torchscience::impl::special_functions::hyperbolic_cosine_integral_chi_backward(x);
}

TORCHSCIENCE_UNARY_CPU_KERNEL(hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_CPU_KERNEL_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::cpu::special_functions
