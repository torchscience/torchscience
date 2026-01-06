#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

class PoissonCdfFunction : public torch::autograd::Function<PoissonCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& k,
      const at::Tensor& rate
  ) {
    ctx->save_for_backward({k, rate});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::poisson_cumulative_distribution", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(k, rate);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor k = saved[0];
    at::Tensor rate = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::poisson_cumulative_distribution_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, k, rate);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor poisson_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& rate
) {
  return PoissonCdfFunction::apply(k, rate);
}

class PoissonPmfFunction : public torch::autograd::Function<PoissonPmfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& k,
      const at::Tensor& rate
  ) {
    ctx->save_for_backward({k, rate});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::poisson_pmf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(k, rate);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor k = saved[0];
    at::Tensor rate = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::poisson_pmf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, k, rate);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor poisson_pmf(
    const at::Tensor& k,
    const at::Tensor& rate
) {
  return PoissonPmfFunction::apply(k, rate);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("poisson_cumulative_distribution", &torchscience::autograd::probability::poisson_cumulative_distribution);
  m.impl("poisson_pmf", &torchscience::autograd::probability::poisson_pmf);
}
