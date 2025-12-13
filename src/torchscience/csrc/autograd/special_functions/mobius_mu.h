#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(MobiusMu, mobius_mu)
TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(mobius_mu)

} // namespace torchscience::autograd::special_functions
