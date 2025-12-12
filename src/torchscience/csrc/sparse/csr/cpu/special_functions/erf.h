#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(erf)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(erf)

} // namespace torchscience::sparse::csr::cpu::special_functions
