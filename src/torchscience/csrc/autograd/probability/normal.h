#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

/**
 * Autograd function for normal CDF backward pass.
 * This enables second-order gradients by making the backward pass differentiable.
 */
class NormalCdfBackwardFunction : public torch::autograd::Function<NormalCdfBackwardFunction> {
public:
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& grad_output,
      const at::Tensor& x,
      const at::Tensor& loc,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({grad_output, x, loc, scale});

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::normal_cdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, loc, scale);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor grad_output = saved[0];
    at::Tensor x = saved[1];
    at::Tensor loc = saved[2];
    at::Tensor scale = saved[3];

    at::Tensor gg_x = grad_outputs[0];
    at::Tensor gg_loc = grad_outputs[1];
    at::Tensor gg_scale = grad_outputs[2];

    // Handle undefined gradients
    if (!gg_x.defined()) gg_x = at::zeros_like(x);
    if (!gg_loc.defined()) gg_loc = at::zeros_like(loc);
    if (!gg_scale.defined()) gg_scale = at::zeros_like(scale);

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::normal_cdf_backward_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(gg_x, gg_loc, gg_scale, grad_output, x, loc, scale);

    // Returns: grad_grad_output, grad_x, grad_loc, grad_scale
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result)};
  }
};

/**
 * Autograd function for normal CDF.
 */
class NormalCdfFunction : public torch::autograd::Function<NormalCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& loc,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({x, loc, scale});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::normal_cdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, loc, scale);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor loc = saved[1];
    at::Tensor scale = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    // Use NormalCdfBackwardFunction to enable second-order gradients
    auto result = NormalCdfBackwardFunction::apply(grad_output, x, loc, scale);

    return result;
  }
};

inline at::Tensor normal_cdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale
) {
  return NormalCdfFunction::apply(x, loc, scale);
}

/**
 * Autograd function for normal PDF.
 */
class NormalPdfFunction : public torch::autograd::Function<NormalPdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& loc,
      const at::Tensor& scale
  ) {
    ctx->save_for_backward({x, loc, scale});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::normal_pdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, loc, scale);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor loc = saved[1];
    at::Tensor scale = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::normal_pdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, loc, scale);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor normal_pdf(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale
) {
  return NormalPdfFunction::apply(x, loc, scale);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("normal_cdf", &torchscience::autograd::probability::normal_cdf);
  m.impl("normal_pdf", &torchscience::autograd::probability::normal_pdf);
}
