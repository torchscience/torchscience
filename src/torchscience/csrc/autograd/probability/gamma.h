#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

class GammaCdfFunction : public torch::autograd::Function<GammaCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& shape,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({x, shape, scale});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_cdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, shape, scale);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor shape = saved[1];
    at::Tensor scale = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_cdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, shape, scale);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor gamma_cdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale
) {
  return GammaCdfFunction::apply(x, shape, scale);
}

class GammaPdfFunction : public torch::autograd::Function<GammaPdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& shape,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({x, shape, scale});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_pdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, shape, scale);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor shape = saved[1];
    at::Tensor scale = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_pdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, shape, scale);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor gamma_pdf(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale
) {
  return GammaPdfFunction::apply(x, shape, scale);
}

class GammaPpfFunction : public torch::autograd::Function<GammaPpfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& shape,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({p, shape, scale});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_ppf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(p, shape, scale);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor shape = saved[1];
    at::Tensor scale = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gamma_ppf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, p, shape, scale);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor gamma_ppf(
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale
) {
  return GammaPpfFunction::apply(p, shape, scale);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("gamma_cdf", &torchscience::autograd::probability::gamma_cdf);
  m.impl("gamma_pdf", &torchscience::autograd::probability::gamma_pdf);
  m.impl("gamma_ppf", &torchscience::autograd::probability::gamma_ppf);
}
