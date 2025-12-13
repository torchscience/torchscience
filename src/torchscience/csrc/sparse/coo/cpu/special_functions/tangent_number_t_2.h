#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(tangent_number_t_2)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(tangent_number_t_2)

} // namespace torchscience::sparse::coo::cpu::special_functions
