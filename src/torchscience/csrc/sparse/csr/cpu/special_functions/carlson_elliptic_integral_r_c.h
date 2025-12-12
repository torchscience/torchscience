#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL(carlson_elliptic_integral_r_c, x, y)
TORCHSCIENCE_BINARY_SPARSE_CSR_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_c)

} // namespace torchscience::sparse::csr::cpu::special_functions
