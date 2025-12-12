#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(complete_elliptic_integral_e)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(complete_elliptic_integral_e)

} // namespace torchscience::sparse::csr::cpu::special_functions
