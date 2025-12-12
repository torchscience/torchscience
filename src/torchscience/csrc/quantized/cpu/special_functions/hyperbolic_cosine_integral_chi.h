#pragma once

#include <torchscience/csrc/impl/special_functions/hyperbolic_cosine_integral_chi.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t hyperbolic_cosine_integral_chi(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_cosine_integral_chi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(hyperbolic_cosine_integral_chi)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(hyperbolic_cosine_integral_chi)

} // namespace torchscience::quantized::cpu::special_functions
