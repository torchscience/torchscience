#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(RiemannZeta, riemann_zeta)
TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(riemann_zeta)

} // namespace torchscience::autograd::special_functions
