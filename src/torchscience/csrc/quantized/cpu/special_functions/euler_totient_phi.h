#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_totient_phi.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t euler_totient_phi(scalar_t x) {
    return torchscience::impl::special_functions::euler_totient_phi(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(euler_totient_phi)
TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(euler_totient_phi)

} // namespace torchscience::quantized::cpu::special_functions
