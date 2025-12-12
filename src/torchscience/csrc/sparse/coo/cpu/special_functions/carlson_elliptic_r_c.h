#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(carlson_elliptic_r_c, x, y)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(carlson_elliptic_r_c)

} // namespace torchscience::sparse::coo::cpu::special_functions
