#pragma once

#include <torchscience/csrc/autocast/macros.h>

namespace torchscience::autocast::special_functions {

TORCHSCIENCE_BINARY_AUTOCAST(bessel_y_derivative, nu, x)

TORCHSCIENCE_BINARY_AUTOCAST_IMPL(bessel_y_derivative)

} // namespace torchscience::autocast::special_functions
