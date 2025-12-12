#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(sine_integral_si)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(sine_integral_si)

} // namespace torchscience::sparse::csr::cpu::special_functions
