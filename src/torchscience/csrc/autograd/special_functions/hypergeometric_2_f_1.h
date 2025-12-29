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

    at::AutoDispatchBelowAutograd guard;
    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::hypergeometric_2_f_1_backward", "")
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();

    auto [grad_a, grad_b, grad_c, grad_z] = op.call(grad, a, b, c, z);

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
  m.impl("hypergeometric_2_f_1",
         torchscience::autograd::hypergeometric_2_f_1_autograd);
}
