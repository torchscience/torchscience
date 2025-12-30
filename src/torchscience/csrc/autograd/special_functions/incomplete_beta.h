#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

class IncompleteBetaBackward : public torch::autograd::Function<IncompleteBetaBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext *context,
    const at::Tensor &grad_output,
    const at::Tensor &x,
    const at::Tensor &a,
    const at::Tensor &b,
    bool x_requires_grad,
    bool a_requires_grad,
    bool b_requires_grad
  ) {
    context->save_for_backward({grad_output, x, a, b});

    context->saved_data["x_requires_grad"] = x_requires_grad;
    context->saved_data["a_requires_grad"] = a_requires_grad;
    context->saved_data["b_requires_grad"] = b_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::incomplete_beta_backward", "")
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &
      )>();

    auto [grad_x, grad_a, grad_b] = op.call(grad_output, x, a, b);

    return {grad_x, grad_a, grad_b};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext *context,
    const std::vector<at::Tensor> &grad_outputs
  ) {
    auto saved = context->get_saved_variables();

    bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
    bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();
    bool b_requires_grad = context->saved_data["b_requires_grad"].toBool();

    at::Tensor gg_x = grad_outputs[0];
    at::Tensor gg_a = grad_outputs[1];
    at::Tensor gg_b = grad_outputs[2];

    if (!gg_x.defined() && !gg_a.defined() && !gg_b.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
              at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::incomplete_beta_backward_backward", "")
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &
      )>();

    auto [gg_out, new_grad_x, new_grad_a, new_grad_b] = op.call(
      gg_x.defined() ? gg_x : at::zeros_like(saved[0]),
      gg_a.defined() ? gg_a : at::zeros_like(saved[0]),
      gg_b.defined() ? gg_b : at::zeros_like(saved[0]),
      saved[0],
      saved[1],
      saved[2],
      saved[3]
    );

    return {
      gg_out,
      x_requires_grad ? new_grad_x : at::Tensor(),
      a_requires_grad ? new_grad_a : at::Tensor(),
      b_requires_grad ? new_grad_b : at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()
    };
  }
};

class IncompleteBetaForward : public torch::autograd::Function<IncompleteBetaForward> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext *context,
    const at::Tensor &x,
    const at::Tensor &a,
    const at::Tensor &b
  ) {
    context->save_for_backward({x, a, b});

    auto is_differentiable = [](const at::Tensor &t) {
      return t.requires_grad() && (isFloatingType(t.scalar_type()) || isComplexType(t.scalar_type()));
    };

    context->saved_data["x_requires_grad"] = is_differentiable(x);
    context->saved_data["a_requires_grad"] = is_differentiable(a);
    context->saved_data["b_requires_grad"] = is_differentiable(b);

    at::AutoDispatchBelowAutograd guard;

    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::incomplete_beta", "")
      .typed<at::Tensor(
        const at::Tensor &,
        const at::Tensor &,
        const at::Tensor &
      )>();

    return op.call(x, a, b);
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext *context,
    const torch::autograd::variable_list &grad_outputs
  ) {
    auto saved = context->get_saved_variables();

    bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
    bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();
    bool b_requires_grad = context->saved_data["b_requires_grad"].toBool();

    if (!x_requires_grad && !a_requires_grad && !b_requires_grad) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    auto grads = IncompleteBetaBackward::apply(
      grad_outputs[0],
      saved[0],
      saved[1],
      saved[2],
      x_requires_grad,
      a_requires_grad,
      b_requires_grad
    );

    return {
      x_requires_grad ? grads[0] : at::Tensor(),
      a_requires_grad ? grads[1] : at::Tensor(),
      b_requires_grad ? grads[2] : at::Tensor()
    };
  }
};

inline at::Tensor incomplete_beta(
  const at::Tensor &x,
  const at::Tensor &a,
  const at::Tensor &b
) {
  return IncompleteBetaForward::apply(x, a, b);
}

} // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("incomplete_beta", torchscience::autograd::incomplete_beta);
}
