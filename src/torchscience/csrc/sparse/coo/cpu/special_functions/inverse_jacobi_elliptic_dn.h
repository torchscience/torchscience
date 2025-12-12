#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(inverse_jacobi_elliptic_dn, x, k)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(inverse_jacobi_elliptic_dn)

} // namespace torchscience::sparse::coo::cpu::special_functions
