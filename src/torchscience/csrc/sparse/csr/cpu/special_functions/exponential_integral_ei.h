#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(exponential_integral_ei)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(exponential_integral_ei)

} // namespace torchscience::sparse::csr::cpu::special_functions
