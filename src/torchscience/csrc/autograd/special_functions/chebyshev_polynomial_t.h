#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

class ChebyshevPolynomialTBackward
    : public torch::autograd::Function<ChebyshevPolynomialTBackward> {
public:
  static std::vector<at::Tensor>
  forward(torch::autograd::AutogradContext *ctx, const at::Tensor &grad_output,
          const at::Tensor &x, const at::Tensor &n, bool x_requires_grad) {
    ctx->save_for_backward({grad_output, x, n});
    ctx->saved_data["x_requires_grad"] = x_requires_grad;

    at::AutoDispatchBelowAutograd guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_backward",
                               "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    auto [grad_x, grad_n] = op.call(grad_output, x, n);
    return {grad_x, grad_n};
  }

  static std::vector<at::Tensor>
  backward(torch::autograd::AutogradContext *ctx,
           const std::vector<at::Tensor> &grad_outputs) {
    auto saved = ctx->get_saved_variables();
    bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();

    if (!grad_outputs[0].defined() && !grad_outputs[1].defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;
    at::Tensor gg_x =
        grad_outputs[0].defined() ? grad_outputs[0] : at::Tensor();
    at::Tensor gg_n =
        grad_outputs[1].defined() ? grad_outputs[1] : at::Tensor();

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "torchscience::chebyshev_polynomial_t_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &, const at::Tensor &)>();
    auto [gg_out, new_grad_x, new_grad_n] =
        op.call(gg_x, gg_n, saved[0], saved[1], saved[2]);
    return {gg_out, new_grad_x, new_grad_n, at::Tensor()};
  }
};

class ChebyshevPolynomialTForward
    : public torch::autograd::Function<ChebyshevPolynomialTForward> {
public:
  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            const at::Tensor &x, const at::Tensor &n) {
    ctx->save_for_backward({x, n});
    ctx->saved_data["x_requires_grad"] =
        x.requires_grad() && (at::isFloatingType(x.scalar_type()) ||
                              at::isComplexType(x.scalar_type()));

    at::AutoDispatchBelowAutograd guard;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_t", "")
            .typed<at::Tensor(const at::Tensor &, const at::Tensor &)>();
    return op.call(x, n);
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           const torch::autograd::variable_list &grad_outputs) {
    auto saved = ctx->get_saved_variables();
    bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();
    auto grads = ChebyshevPolynomialTBackward::apply(grad_outputs[0], saved[0],
                                                     saved[1], x_requires_grad);
    return {x_requires_grad ? grads[0] : at::Tensor(), grads[1]};
  }
};

inline at::Tensor chebyshev_polynomial_t(const at::Tensor &x,
                                         const at::Tensor &n) {
  return ChebyshevPolynomialTForward::apply(x, n);
}

} // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
  module.impl("chebyshev_polynomial_t",
              torchscience::autograd::chebyshev_polynomial_t);
}
