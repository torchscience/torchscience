#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(hyperbolic_sine_integral_shi)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(hyperbolic_sine_integral_shi)

} // namespace torchscience::sparse::coo::cpu::special_functions
