#pragma once

#include <torchscience/csrc/impl/special_functions/cosine_integral_ci.h>
#include <torchscience/csrc/quantized/cpu/macros.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t cosine_integral_ci(scalar_t x) {
  return torchscience::impl::special_functions::cosine_integral_ci(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(cosine_integral_ci)

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(cosine_integral_ci)

} // namespace torchscience::quantized::cpu::special_functions
