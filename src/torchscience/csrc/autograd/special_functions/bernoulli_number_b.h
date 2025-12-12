#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(BernoulliNumberB, bernoulli_number_b)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(bernoulli_number_b)

} // namespace torchscience::autograd::special_functions
