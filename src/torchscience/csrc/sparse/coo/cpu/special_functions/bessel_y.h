#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(bessel_y, nu, x)
TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(bessel_y)

} // namespace torchscience::sparse::coo::cpu::special_functions
