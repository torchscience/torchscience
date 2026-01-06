#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

class BetaCdfFunction : public torch::autograd::Function<BetaCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& a,
      const at::Tensor& b
  ) {
    ctx->save_for_backward({x, a, b});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_cdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, a, b);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor a = saved[1];
    at::Tensor b = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_cdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, a, b);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor beta_cdf(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
  return BetaCdfFunction::apply(x, a, b);
}

class BetaPdfFunction : public torch::autograd::Function<BetaPdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& a,
      const at::Tensor& b
  ) {
    ctx->save_for_backward({x, a, b});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_pdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, a, b);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor a = saved[1];
    at::Tensor b = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_pdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, a, b);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor beta_pdf(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
  return BetaPdfFunction::apply(x, a, b);
}

class BetaPpfFunction : public torch::autograd::Function<BetaPpfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& a,
      const at::Tensor& b
  ) {
    ctx->save_for_backward({p, a, b});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_ppf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(p, a, b);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor a = saved[1];
    at::Tensor b = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::beta_ppf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, p, a, b);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor beta_ppf(
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b
) {
  return BetaPpfFunction::apply(p, a, b);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("beta_cdf", &torchscience::autograd::probability::beta_cdf);
  m.impl("beta_pdf", &torchscience::autograd::probability::beta_pdf);
  m.impl("beta_ppf", &torchscience::autograd::probability::beta_ppf);
}
