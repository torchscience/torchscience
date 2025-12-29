#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

class Hypergeometric2F1Function : public torch::autograd::Function<Hypergeometric2F1Function> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &c,
    const at::Tensor &z
  ) {
    ctx->save_for_backward({a, b, c, z});
    ctx->saved_data["needs_param_grad"] = a.requires_grad() || b.requires_grad() || c.requires_grad();

    at::AutoDispatchBelowAutograd guard;
    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::hypergeometric_2_f_1", "")
      .typed<at::Tensor(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(a, b, c, z);
  }

  static torch::autograd::tensor_list backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto c = saved[2];
    auto z = saved[3];
    auto grad = grad_outputs[0];

    bool needs_param_grad = ctx->saved_data["needs_param_grad"].toBool();

    // d/dz 2F1(a,b;c;z) = (ab/c) * 2F1(a+1, b+1; c+1; z)
    // This is differentiable because it calls forward again
    auto dz_coef = a * b / c;
    auto f_shifted = Hypergeometric2F1Function::apply(a + 1, b + 1, c + 1, z);
    auto grad_z = grad * dz_coef * f_shifted;

    at::Tensor grad_a, grad_b, grad_c;

    if (needs_param_grad) {
      // For differentiable parameter gradients in the |z| < 0.5 region,
      // use finite differences with autograd-enabled forward
      // This allows second derivatives to work via torch autograd
      auto eps = at::full_like(a, 1e-5);

      // d/da using central difference (differentiable)
      auto f_a_plus = Hypergeometric2F1Function::apply(a + eps, b, c, z);
      auto f_a_minus = Hypergeometric2F1Function::apply(a - eps, b, c, z);
      grad_a = grad * (f_a_plus - f_a_minus) / (2 * eps);

      // d/db
      auto f_b_plus = Hypergeometric2F1Function::apply(a, b + eps, c, z);
      auto f_b_minus = Hypergeometric2F1Function::apply(a, b - eps, c, z);
      grad_b = grad * (f_b_plus - f_b_minus) / (2 * eps);

      // d/dc
      auto f_c_plus = Hypergeometric2F1Function::apply(a, b, c + eps, z);
      auto f_c_minus = Hypergeometric2F1Function::apply(a, b, c - eps, z);
      grad_c = grad * (f_c_plus - f_c_minus) / (2 * eps);
    } else {
      grad_a = at::zeros_like(a);
      grad_b = at::zeros_like(b);
      grad_c = at::zeros_like(c);
    }

    return {grad_a, grad_b, grad_c, grad_z};
  }
};

inline at::Tensor hypergeometric_2_f_1_autograd(
  const at::Tensor &a,
  const at::Tensor &b,
  const at::Tensor &c,
  const at::Tensor &z
) {
  return Hypergeometric2F1Function::apply(a, b, c, z);
}

} // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("hypergeometric_2_f_1", torchscience::autograd::hypergeometric_2_f_1_autograd);
}
