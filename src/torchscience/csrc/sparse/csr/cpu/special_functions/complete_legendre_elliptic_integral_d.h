#pragma once

#include <torchscience/csrc/sparse/csr/cpu/macros.h>

namespace torchscience::sparse::csr::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL(complete_legendre_elliptic_integral_d)

TORCHSCIENCE_UNARY_SPARSE_CSR_CPU_KERNEL_IMPL(complete_legendre_elliptic_integral_d)

} // namespace torchscience::sparse::csr::cpu::special_functions
