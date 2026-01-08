#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::window_function {

namespace {

inline at::TensorOptions build_meta_options(
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout
) {
  return at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
}

}  // anonymous namespace

// =============================================================================
// Parameterless windows - all have the same signature
// =============================================================================

#define TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(name)                     \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  (void)device;  /* Meta ignores device */                                      \
  auto options = build_meta_options(dtype, layout);                             \
  return at::empty({n}, options).requires_grad_(requires_grad);                 \
}

TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(hamming)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_hamming)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(blackman)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_blackman)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(bartlett)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_bartlett)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(cosine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_cosine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(nuttall)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_nuttall)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(triangular)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_triangular)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(welch)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_welch)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(parzen)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_parzen)

#undef TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW

// =============================================================================
// Parameterized windows: Gaussian
// =============================================================================

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(std_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window(n, std_input, dtype, layout, device);
}

inline at::Tensor gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(std_input, at::TensorOptions().dtype(std_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward(grad_output, output, n, std_input);
}

// =============================================================================
// Parameterized windows: General Hamming
// =============================================================================

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window(n, alpha_input, dtype, layout, device);
}

inline at::Tensor general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(alpha_input, at::TensorOptions().dtype(alpha_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward(grad_output, output, n, alpha_input);
}

// =============================================================================
// Parameterized windows: General Cosine
// =============================================================================

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(coeffs_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window(n, coeffs_input, dtype, layout, device);
}

inline at::Tensor general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(coeffs_input, at::TensorOptions().dtype(coeffs_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward(grad_output, output, n, coeffs_input);
}

}  // namespace torchscience::meta::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("hann_window", torchscience::meta::window_function::hann_window);
  m.impl("periodic_hann_window", torchscience::meta::window_function::periodic_hann_window);
  m.impl("hamming_window", torchscience::meta::window_function::hamming_window);
  m.impl("periodic_hamming_window", torchscience::meta::window_function::periodic_hamming_window);
  m.impl("blackman_window", torchscience::meta::window_function::blackman_window);
  m.impl("periodic_blackman_window", torchscience::meta::window_function::periodic_blackman_window);
  m.impl("bartlett_window", torchscience::meta::window_function::bartlett_window);
  m.impl("periodic_bartlett_window", torchscience::meta::window_function::periodic_bartlett_window);
  m.impl("cosine_window", torchscience::meta::window_function::cosine_window);
  m.impl("periodic_cosine_window", torchscience::meta::window_function::periodic_cosine_window);
  m.impl("nuttall_window", torchscience::meta::window_function::nuttall_window);
  m.impl("periodic_nuttall_window", torchscience::meta::window_function::periodic_nuttall_window);
  m.impl("triangular_window", torchscience::meta::window_function::triangular_window);
  m.impl("periodic_triangular_window", torchscience::meta::window_function::periodic_triangular_window);
  m.impl("welch_window", torchscience::meta::window_function::welch_window);
  m.impl("periodic_welch_window", torchscience::meta::window_function::periodic_welch_window);
  m.impl("parzen_window", torchscience::meta::window_function::parzen_window);
  m.impl("periodic_parzen_window", torchscience::meta::window_function::periodic_parzen_window);

  m.impl("gaussian_window", torchscience::meta::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::meta::window_function::periodic_gaussian_window);
  m.impl("gaussian_window_backward", torchscience::meta::window_function::gaussian_window_backward);
  m.impl("periodic_gaussian_window_backward", torchscience::meta::window_function::periodic_gaussian_window_backward);

  m.impl("general_hamming_window", torchscience::meta::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::meta::window_function::periodic_general_hamming_window);
  m.impl("general_hamming_window_backward", torchscience::meta::window_function::general_hamming_window_backward);
  m.impl("periodic_general_hamming_window_backward", torchscience::meta::window_function::periodic_general_hamming_window_backward);

  m.impl("general_cosine_window", torchscience::meta::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::meta::window_function::periodic_general_cosine_window);
  m.impl("general_cosine_window_backward", torchscience::meta::window_function::general_cosine_window_backward);
  m.impl("periodic_general_cosine_window_backward", torchscience::meta::window_function::periodic_general_cosine_window_backward);
}
