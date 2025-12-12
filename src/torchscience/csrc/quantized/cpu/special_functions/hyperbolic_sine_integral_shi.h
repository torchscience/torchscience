#pragma once

#include <torchscience/csrc/impl/special_functions/hyperbolic_sine_integral_shi.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t hyperbolic_sine_integral_shi(scalar_t x) {
  return torchscience::impl::special_functions::hyperbolic_sine_integral_shi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::quantized::cpu::special_functions
