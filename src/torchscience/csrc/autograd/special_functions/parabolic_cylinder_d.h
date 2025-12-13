#pragma once

#include <torchscience/csrc/autograd/macros.h>

namespace torchscience::autograd::special_functions {

TORCHSCIENCE_BINARY_AUTOGRAD(ParabolicCylinderDFunction, parabolic_cylinder_d, nu, z)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(parabolic_cylinder_d)

} // namespace torchscience::autograd::special_functions
