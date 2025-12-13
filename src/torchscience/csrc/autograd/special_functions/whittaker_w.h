#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(WhittakerWFunction, whittaker_w, kappa, mu, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(whittaker_w)

} // namespace torchscience::autograd::special_functions
