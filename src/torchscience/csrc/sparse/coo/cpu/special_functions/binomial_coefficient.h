#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(binomial_coefficient, n, k)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(binomial_coefficient)

} // namespace torchscience::sparse::coo::cpu::special_functions
