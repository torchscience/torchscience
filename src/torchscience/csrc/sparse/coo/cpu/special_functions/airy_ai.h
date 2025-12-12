#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(airy_ai)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(airy_ai)

} // namespace torchscience::sparse::coo::cpu::special_functions
