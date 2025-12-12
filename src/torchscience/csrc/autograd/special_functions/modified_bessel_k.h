#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ModifiedBesselKFunction, modified_bessel_k, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(modified_bessel_k)

} // namespace torchscience::autograd::special_functions
