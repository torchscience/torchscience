#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(bernoulli_number_b)

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(bernoulli_number_b)

} // namespace torchscience::sparse::coo::cpu::special_functions
