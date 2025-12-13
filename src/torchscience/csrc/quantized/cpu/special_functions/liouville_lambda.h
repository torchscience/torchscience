#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/liouville_lambda.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t liouville_lambda(scalar_t x) {
    return torchscience::impl::special_functions::liouville_lambda(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(liouville_lambda)
TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(liouville_lambda)

} // namespace torchscience::quantized::cpu::special_functions
