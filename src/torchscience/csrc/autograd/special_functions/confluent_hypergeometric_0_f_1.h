#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ConfluentHypergeometric0F1Function, confluent_hypergeometric_0_f_1, b, z)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(confluent_hypergeometric_0_f_1)

} // namespace torchscience::autograd::special_functions
