#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(bulirsch_elliptic_integral_el1, x, kc)

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(bulirsch_elliptic_integral_el1)

} // namespace torchscience::sparse::csr::cpu::special_functions
