#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(exponential_integral_e_1)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(exponential_integral_e_1)

} // namespace torchscience::sparse::coo::cpu::special_functions
