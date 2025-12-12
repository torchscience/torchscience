#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(BesselYFunction, bessel_y, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bessel_y)
