#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_number_e.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t euler_number_e(scalar_t x) {
    return torchscience::impl::special_functions::euler_number_e(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(euler_number_e)
TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(euler_number_e)

} // namespace torchscience::quantized::cpu::special_functions
