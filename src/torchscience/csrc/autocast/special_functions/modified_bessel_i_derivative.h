#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(modified_bessel_i_derivative, nu, x)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(modified_bessel_i_derivative)

} // namespace torchscience::autocast::special_functions
