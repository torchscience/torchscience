#pragma once

#include <cmath>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

namespace torchscience::cpu {

namespace {

template <typename T> T chebyshev_polynomial_t_kernel(T x, T n) {
  if (std::abs(x) <= T(1)) {
    return std::cos(n * std::acos(x));
  }
  if (x > T(1)) {
    return std::cosh(n * std::acosh(x));
  }
  T sign = (static_cast<int>(n) % 2 == 0) ? T(1) : T(-1);
  return sign * std::cosh(n * std::acosh(-x));
}

} // anonymous namespace

inline at::Tensor chebyshev_polynomial_t_forward(
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_cpu",
    [&] {
      at::native::cpu_kernel(
        iterator,
        [](
          scalar_t x,
          scalar_t n
        ) -> scalar_t {
          return chebyshev_polynomial_t_kernel(x, n);
        }
      );
    }
  );

  return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor grad_x;
  at::Tensor grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_x)
    .add_output(grad_n)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [](
          scalar_t g,
          scalar_t x,
          scalar_t n
        ) -> std::tuple<scalar_t, scalar_t> {
          scalar_t eps = scalar_t(1e-6);

          scalar_t grad_x = g * n * (chebyshev_polynomial_t_kernel(x + eps, n) - chebyshev_polynomial_t_kernel(x - eps, n)) / (scalar_t(2) * eps);

          return {
            grad_x,
            scalar_t(0)
          };
        }
      );
    }
  );

  return {
    iterator.output(0),
    iterator.output(1)
  };
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
  const at::Tensor &gg_x,
  const at::Tensor &gg_n,
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  bool has_gg_x = gg_x.defined();
  bool has_gg_n = gg_n.defined();

  if (!has_gg_x && !has_gg_n) {
    return {
      at::Tensor(),
      at::Tensor(),
      at::Tensor()
    };
  }

  at::Tensor gg_x_safe;
  at::Tensor gg_n_safe;

  if (has_gg_x) {
    gg_x_safe = gg_x;
  } else {
    gg_x_safe = zeros_like(grad);
  }

  if (has_gg_n) {
    gg_n_safe = gg_n;
  } else {
    gg_n_safe = zeros_like(grad);
  }

  at::Tensor gg_out;
  at::Tensor new_grad_x;
  at::Tensor new_grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_out)
    .add_output(new_grad_x)
    .add_output(new_grad_n)
    .add_const_input(gg_x_safe)
    .add_const_input(gg_n_safe)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [has_gg_x](
          scalar_t gg_x,
          scalar_t gg_n,
          scalar_t g,
          scalar_t x,
          scalar_t n
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
          auto eps = scalar_t(1e-5);
          scalar_t d2 = (chebyshev_polynomial_t_kernel(x + eps, n) - scalar_t(2) * chebyshev_polynomial_t_kernel(x, n) + chebyshev_polynomial_t_kernel(x - eps, n)) / (eps * eps);

          scalar_t gg_out = has_gg_x ? gg_x * n * (chebyshev_polynomial_t_kernel(x + eps, n) - chebyshev_polynomial_t_kernel(x - eps, n)) / (scalar_t(2) * eps) : scalar_t(0);

          scalar_t new_grad_x = has_gg_x ? gg_x * g * n * n * d2 : scalar_t(0);

          return {
            gg_out,
            new_grad_x,
            scalar_t(0)
          };
        }
      );
    }
  );

  return {
    iterator.output(0),
    iterator.output(1),
    iterator.output(2)
  };
}

} // namespace torchscience::cpu

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
  module.impl(
    "chebyshev_polynomial_t",
    torchscience::cpu::chebyshev_polynomial_t_forward
  );

  module.impl(
    "chebyshev_polynomial_t_backward",
    torchscience::cpu::chebyshev_polynomial_t_backward
  );

  module.impl(
    "chebyshev_polynomial_t_backward_backward",
    torchscience::cpu::chebyshev_polynomial_t_backward_backward
  );
}
