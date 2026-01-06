#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

/**
 * Autograd function for F CDF.
 */
class FCdfFunction : public torch::autograd::Function<FCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& dfn,
      const at::Tensor& dfd
  ) {
    ctx->save_for_backward({x, dfn, dfd});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_cdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, dfn, dfd);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor dfn = saved[1];
    at::Tensor dfd = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_cdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, dfn, dfd);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor f_cdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd
) {
  return FCdfFunction::apply(x, dfn, dfd);
}

/**
 * Autograd function for F PDF.
 */
class FPdfFunction : public torch::autograd::Function<FPdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& dfn,
      const at::Tensor& dfd
  ) {
    ctx->save_for_backward({x, dfn, dfd});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_pdf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, dfn, dfd);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor dfn = saved[1];
    at::Tensor dfd = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_pdf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, dfn, dfd);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor f_pdf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd
) {
  return FPdfFunction::apply(x, dfn, dfd);
}

/**
 * Autograd function for F PPF.
 */
class FPpfFunction : public torch::autograd::Function<FPpfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& p,
      const at::Tensor& dfn,
      const at::Tensor& dfd
  ) {
    ctx->save_for_backward({p, dfn, dfd});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_ppf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(p, dfn, dfd);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor p = saved[0];
    at::Tensor dfn = saved[1];
    at::Tensor dfd = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_ppf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, p, dfn, dfd);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor f_ppf(
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd
) {
  return FPpfFunction::apply(p, dfn, dfd);
}

/**
 * Autograd function for F SF.
 */
class FSfFunction : public torch::autograd::Function<FSfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& dfn,
      const at::Tensor& dfd
  ) {
    ctx->save_for_backward({x, dfn, dfd});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_sf", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(x, dfn, dfd);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor x = saved[0];
    at::Tensor dfn = saved[1];
    at::Tensor dfd = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::f_sf_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, x, dfn, dfd);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor f_sf(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd
) {
  return FSfFunction::apply(x, dfn, dfd);
}

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("f_cdf", &f_cdf);
  m.impl("f_pdf", &f_pdf);
  m.impl("f_ppf", &f_ppf);
  m.impl("f_sf", &f_sf);
}

}  // namespace torchscience::autograd::probability
