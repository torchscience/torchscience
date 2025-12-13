#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(WhittakerMFunction, whittaker_m, kappa, mu, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(whittaker_m)

} // namespace torchscience::autograd::special_functions
