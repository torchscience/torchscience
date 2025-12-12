#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(sinc_pi)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(sinc_pi)

} // namespace torchscience::sparse::coo::cpu::special_functions
