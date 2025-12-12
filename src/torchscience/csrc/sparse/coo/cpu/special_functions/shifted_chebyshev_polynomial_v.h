#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(shifted_chebyshev_polynomial_v, n, x)

} // namespace torchscience::sparse::coo::cpu::special_functions
