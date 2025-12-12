#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(JacobiAmplitudeAmFunction, jacobi_amplitude_am, u, k)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(jacobi_amplitude_am)

} // namespace torchscience::autograd::special_functions
