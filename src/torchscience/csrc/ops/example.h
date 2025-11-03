#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace science {
namespace ops {
SCIENCE_API
at::Tensor example(
    const at::Tensor& input,
    int64_t foo,
    int64_t bar,
    int64_t baz
);

SCIENCE_API
at::Tensor example_symint(
    const at::Tensor& input,
    c10::SymInt foo,
    c10::SymInt bar,
    c10::SymInt baz
);

namespace detail {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_example_backward(
    const at::Tensor& gradient,
    const at::Tensor& input,
    int64_t foo,
    int64_t bar,
    int64_t baz
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_example_backward_symint(
    const at::Tensor& gradient,
    c10::SymInt foo,
    c10::SymInt bar,
    c10::SymInt baz
);
} // namespace detail
} // namespace ops
} // namespace science
