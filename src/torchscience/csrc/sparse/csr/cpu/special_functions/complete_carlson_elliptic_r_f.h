#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(complete_carlson_elliptic_r_f, x, y)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(complete_carlson_elliptic_r_f)

} // namespace torchscience::sparse::csr::cpu::special_functions
