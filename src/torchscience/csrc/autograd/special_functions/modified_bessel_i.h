#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ModifiedBesselIFunction, modified_bessel_i, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(modified_bessel_i)

} // namespace torchscience::autograd::special_functions
