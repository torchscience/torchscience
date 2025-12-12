#pragma once

#include <torchscience/csrc/meta/macros.h>

namespace torchscience::meta::special_functions {

TORCHSCIENCE_BINARY_META_KERNEL(jacobi_amplitude_am, u, k)
TORCHSCIENCE_BINARY_META_KERNEL_IMPL(jacobi_amplitude_am)

} // namespace torchscience::meta::special_functions
