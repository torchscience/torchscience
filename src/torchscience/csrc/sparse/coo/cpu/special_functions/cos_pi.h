#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(cos_pi)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(cos_pi)

} // namespace torchscience::sparse::coo::cpu::special_functions
