#pragma once

#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::window_function {

// =============================================================================
// Gaussian Window Autograd
// =============================================================================

class GaussianWindowBackward : public torch::autograd::Function<GaussianWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& std_input,
    bool periodic,
    bool std_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, std_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["std_requires_grad"] = std_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_gaussian_window_backward"
      : "torchscience::gaussian_window_backward";

    auto grad_std = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, std_input);

    return {grad_std};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GaussianWindow : public torch::autograd::Function<GaussianWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& std_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_gaussian_window"
      : "torchscience::gaussian_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, std_input, dtype, layout, device);

    ctx->save_for_backward({output, std_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["std_requires_grad"] = std_input.requires_grad() && at::isFloatingType(std_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool std_requires_grad = ctx->saved_data["std_requires_grad"].toBool();

    at::Tensor grad_std;
    if (std_requires_grad && grad_outputs[0].defined()) {
      auto grads = GaussianWindowBackward::apply(
        grad_outputs[0], saved[0], n, saved[1], periodic, true
      );
      grad_std = grads[0];
    }

    return {at::Tensor(), grad_std, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GaussianWindow::apply(n, std_input, false, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GaussianWindow::apply(n, std_input, true, dtype, layout, device);
}

// =============================================================================
// General Hamming Window Autograd
// =============================================================================

class GeneralHammingWindowBackward : public torch::autograd::Function<GeneralHammingWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    bool alpha_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_hamming_window_backward"
      : "torchscience::general_hamming_window_backward";

    auto grad_alpha = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, alpha_input);

    return {grad_alpha};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GeneralHammingWindow : public torch::autograd::Function<GeneralHammingWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& alpha_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_hamming_window"
      : "torchscience::general_hamming_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, alpha_input, dtype, layout, device);

    ctx->save_for_backward({output, alpha_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["alpha_requires_grad"] = alpha_input.requires_grad() && at::isFloatingType(alpha_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

    at::Tensor grad_alpha;
    if (alpha_requires_grad && grad_outputs[0].defined()) {
      auto grads = GeneralHammingWindowBackward::apply(
        grad_outputs[0], saved[0], n, saved[1], periodic, true
      );
      grad_alpha = grads[0];
    }

    return {at::Tensor(), grad_alpha, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralHammingWindow::apply(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralHammingWindow::apply(n, alpha_input, true, dtype, layout, device);
}

// =============================================================================
// General Cosine Window Autograd
// =============================================================================

class GeneralCosineWindowBackward : public torch::autograd::Function<GeneralCosineWindowBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t n,
    const at::Tensor& coeffs_input,
    bool periodic,
    bool coeffs_requires_grad
  ) {
    ctx->save_for_backward({grad_output, output, coeffs_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_cosine_window_backward"
      : "torchscience::general_cosine_window_backward";

    auto grad_coeffs = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&)>()
      .call(grad_output, output, n, coeffs_input);

    return {grad_coeffs};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad_outputs
  ) {
    // Second-order gradients not implemented for window functions
    (void)ctx;
    (void)grad_outputs;
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

class GeneralCosineWindow : public torch::autograd::Function<GeneralCosineWindow> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    int64_t n,
    const at::Tensor& coeffs_input,
    bool periodic,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device
  ) {
    at::AutoDispatchBelowAutograd guard;

    std::string op_name = periodic
      ? "torchscience::periodic_general_cosine_window"
      : "torchscience::general_cosine_window";

    auto output = c10::Dispatcher::singleton()
      .findSchemaOrThrow(op_name, "")
      .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
      .call(n, coeffs_input, dtype, layout, device);

    ctx->save_for_backward({output, coeffs_input});
    ctx->saved_data["n"] = n;
    ctx->saved_data["periodic"] = periodic;
    ctx->saved_data["coeffs_requires_grad"] = coeffs_input.requires_grad() && at::isFloatingType(coeffs_input.scalar_type());

    return output;
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    const torch::autograd::variable_list& grad_outputs
  ) {
    auto saved = ctx->get_saved_variables();
    int64_t n = ctx->saved_data["n"].toInt();
    bool periodic = ctx->saved_data["periodic"].toBool();
    bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

    at::Tensor grad_coeffs;
    if (coeffs_requires_grad && grad_outputs[0].defined()) {
      auto grads = GeneralCosineWindowBackward::apply(
        grad_outputs[0], saved[0], n, saved[1], periodic, true
      );
      grad_coeffs = grads[0];
    }

    return {at::Tensor(), grad_coeffs, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralCosineWindow::apply(n, coeffs_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return GeneralCosineWindow::apply(n, coeffs_input, true, dtype, layout, device);
}

}  // namespace torchscience::autograd::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("gaussian_window", torchscience::autograd::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::autograd::window_function::periodic_gaussian_window);

  m.impl("general_hamming_window", torchscience::autograd::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::autograd::window_function::periodic_general_hamming_window);

  m.impl("general_cosine_window", torchscience::autograd::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::autograd::window_function::periodic_general_cosine_window);
}
