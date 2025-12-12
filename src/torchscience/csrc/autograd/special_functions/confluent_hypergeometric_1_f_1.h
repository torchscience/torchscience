#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_TERNARY_AUTOGRAD(ConfluentHypergeometric1F1, confluent_hypergeometric_1_f_1, a, b, z)
TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(confluent_hypergeometric_1_f_1)

} // namespace torchscience::autograd::special_functions
