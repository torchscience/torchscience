#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_UNARY_AUTOGRAD(PrimeNumberP, prime_number_p)

TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(prime_number_p)

} // namespace torchscience::autograd::special_functions
