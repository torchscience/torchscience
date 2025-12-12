#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(jacobi_theta_2, z, q)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(jacobi_theta_2)

} // namespace torchscience::sparse::coo::cpu::special_functions
