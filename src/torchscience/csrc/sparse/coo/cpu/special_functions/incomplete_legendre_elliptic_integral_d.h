#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(incomplete_legendre_elliptic_integral_d, phi, k)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(incomplete_legendre_elliptic_integral_d)

} // namespace torchscience::sparse::coo::cpu::special_functions
