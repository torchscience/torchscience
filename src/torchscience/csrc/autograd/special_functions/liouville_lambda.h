#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(LiouvilleLambda, liouville_lambda)
TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(liouville_lambda)

} // namespace torchscience::autograd::special_functions
