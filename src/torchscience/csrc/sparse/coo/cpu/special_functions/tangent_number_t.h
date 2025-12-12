#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(tangent_number_t)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(tangent_number_t)

} // namespace torchscience::sparse::coo::cpu::special_functions
