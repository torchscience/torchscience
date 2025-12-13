#pragma once

#include <torchscience/csrc/quantized/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/mobius_mu.h>

namespace torchscience::quantized::cpu::special_functions {

template <typename scalar_t>
scalar_t mobius_mu(scalar_t x) {
    return torchscience::impl::special_functions::mobius_mu(x);
}

TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL(mobius_mu)
TORCHSCIENCE_UNARY_QUANTIZED_CPU_KERNEL_IMPL(mobius_mu)

} // namespace torchscience::quantized::cpu::special_functions
