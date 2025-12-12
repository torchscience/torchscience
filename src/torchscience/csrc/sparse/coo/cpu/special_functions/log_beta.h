#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL(log_beta, a, b)

TORCHSCIENCE_BINARY_SPARSE_COO_CPU_KERNEL_IMPL(log_beta)

} // namespace torchscience::sparse::coo::cpu::special_functions
