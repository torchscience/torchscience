#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(spherical_bessel_y, n, x)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(spherical_bessel_y)

} // namespace torchscience::sparse::coo::cpu::special_functions
