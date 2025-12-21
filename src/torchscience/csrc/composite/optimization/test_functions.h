#pragma once

#include <torch/extension.h>

#include "../../impl/optimization/test_functions.h"

namespace torchscience::composite::test_functions {

using torchscience::impl::optimization::test_functions::check_rosenbrock_input;

inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    check_rosenbrock_input(x, "rosenbrock");

    // x_i and x_{i+1} components along last dimension
    at::Tensor x_i = x.narrow(-1, 0, x.size(-1) - 1);
    at::Tensor x_i_plus_1 = x.narrow(-1, 1, x.size(-1) - 1);

    // Rosenbrock formula: sum of (a - x_i)^2 + b*(x_{i+1} - x_i^2)^2
    at::Tensor term1 = at::pow(a - x_i, 2);
    at::Tensor term2 = b * at::pow(x_i_plus_1 - at::pow(x_i, 2), 2);

    return at::sum(term1 + term2, -1);
}

}  // namespace torchscience::composite::test_functions

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) {
    module.impl(
        "rosenbrock",
        &torchscience::composite::test_functions::rosenbrock
    );
}
