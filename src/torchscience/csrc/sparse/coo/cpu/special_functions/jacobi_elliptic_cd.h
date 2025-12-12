#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(jacobi_elliptic_cd, u, k)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(jacobi_elliptic_cd)

} // namespace torchscience::sparse::coo::cpu::special_functions
