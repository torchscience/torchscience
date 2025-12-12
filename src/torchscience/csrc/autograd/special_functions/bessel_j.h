#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(BesselJFunction, bessel_j, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(bessel_j)
