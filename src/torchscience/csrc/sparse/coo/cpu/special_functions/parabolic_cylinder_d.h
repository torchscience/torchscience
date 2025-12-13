#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(parabolic_cylinder_d)

} // namespace torchscience::sparse::coo::cpu::special_functions
