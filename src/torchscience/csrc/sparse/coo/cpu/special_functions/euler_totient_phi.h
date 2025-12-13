#pragma once

#include <torchscience/csrc/sparse/coo/cpu/macros.h>
#include <torchscience/csrc/impl/special_functions/euler_totient_phi.h>

namespace torchscience::sparse::coo::cpu::special_functions {

using namespace torchscience::impl::special_functions;

TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL(euler_totient_phi)
TORCHSCIENCE_UNARY_SPARSE_COO_CPU_KERNEL_IMPL(euler_totient_phi)

} // namespace torchscience::sparse::coo::cpu::special_functions
