#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(sine_integral_si)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(sine_integral_si)

} // namespace torchscience::sparse::coo::cpu::special_functions
