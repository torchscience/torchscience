#pragma once

#include "torchscience/csrc/autograd/macros.h"

TORCHSCIENCE_BINARY_AUTOGRAD(ModifiedBesselIFunction, modified_bessel_i, nu, x)
TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(modified_bessel_i)
