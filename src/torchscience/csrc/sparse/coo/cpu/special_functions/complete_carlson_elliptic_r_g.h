#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(complete_carlson_elliptic_r_g, x, y)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(complete_carlson_elliptic_r_g)

} // namespace torchscience::sparse::coo::cpu::special_functions
