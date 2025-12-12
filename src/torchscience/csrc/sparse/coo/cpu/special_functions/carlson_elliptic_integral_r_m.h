#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL(carlson_elliptic_integral_r_m, x, y, z)
TORCHSCIENCE_TERNARY_SPARSE_COO_CPU_KERNEL_IMPL(carlson_elliptic_integral_r_m)

} // namespace torchscience::sparse::coo::cpu::special_functions
