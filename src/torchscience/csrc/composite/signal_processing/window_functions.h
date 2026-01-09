#pragma once

#include <vector>
#include <torch/extension.h>
#include "../../cpu/creation_operators.h"
#include "../../meta/creation_operators.h"

namespace torchscience::composite::window_function {

namespace {

struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) {
        return {n};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, int64_t n) {
        (void)n;  // n == numel for rectangular window
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = scalar_t(1);
        }
    }
};

}  // anonymous namespace

// Composite implementation that routes to appropriate backend based on device
inline at::Tensor rectangular_window(
  int64_t n,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  at::Device target_device = device.value_or(at::kCPU);

  if (target_device.type() == at::kMeta) {
    return torchscience::meta::MetaCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  } else {
    return torchscience::cpu::CPUCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  }
}

// =============================================================================
// Routing macros for parameterless windows
// =============================================================================

#define DEFINE_PARAMETERLESS_WINDOW_ROUTER(name)                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  at::Device target_device = device.value_or(at::kCPU);                         \
  c10::DispatchKeySet ks = target_device.type() == at::kMeta                    \
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)                               \
    : c10::DispatchKeySet(c10::DispatchKey::CPU);                               \
  return c10::Dispatcher::singleton()                                           \
    .findSchemaOrThrow("torchscience::" #name "_window", "")                    \
    .typed<at::Tensor(int64_t, c10::optional<at::ScalarType>,                   \
                      c10::optional<at::Layout>, c10::optional<at::Device>,     \
                      bool)>()                                                  \
    .redispatch(ks, n, dtype, layout, device, requires_grad);                   \
}                                                                               \
                                                                                \
inline at::Tensor periodic_##name##_window(                                     \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  at::Device target_device = device.value_or(at::kCPU);                         \
  c10::DispatchKeySet ks = target_device.type() == at::kMeta                    \
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)                               \
    : c10::DispatchKeySet(c10::DispatchKey::CPU);                               \
  return c10::Dispatcher::singleton()                                           \
    .findSchemaOrThrow("torchscience::periodic_" #name "_window", "")           \
    .typed<at::Tensor(int64_t, c10::optional<at::ScalarType>,                   \
                      c10::optional<at::Layout>, c10::optional<at::Device>,     \
                      bool)>()                                                  \
    .redispatch(ks, n, dtype, layout, device, requires_grad);                   \
}

DEFINE_PARAMETERLESS_WINDOW_ROUTER(hann)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(hamming)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(blackman)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(bartlett)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(cosine)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(nuttall)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(triangular)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(welch)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(parzen)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(blackman_harris)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(flat_top)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(sine)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(bartlett_hann)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(lanczos)

#undef DEFINE_PARAMETERLESS_WINDOW_ROUTER

// =============================================================================
// Routing for parameterized windows
// =============================================================================

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  // For parameterized windows, device from tensor or explicit device param
  at::Device target_device = device.value_or(std_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::gaussian_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, std_input, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(std_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_gaussian_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, std_input, dtype, layout, device);
}

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::general_hamming_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_general_hamming_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(coeffs_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::general_cosine_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, coeffs_input, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(coeffs_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_general_cosine_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, coeffs_input, dtype, layout, device);
}

inline at::Tensor tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::tukey_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor periodic_tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks = target_device.type() == at::kMeta
    ? c10::DispatchKeySet(c10::DispatchKey::Meta)
    : c10::DispatchKeySet(c10::DispatchKey::CPU);
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_tukey_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

} // namespace torchscience::composite::window_function

// =============================================================================
// CompositeExplicitAutograd registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl("rectangular_window", &torchscience::composite::window_function::rectangular_window);

  module.impl("hann_window", &torchscience::composite::window_function::hann_window);
  module.impl("periodic_hann_window", &torchscience::composite::window_function::periodic_hann_window);
  module.impl("hamming_window", &torchscience::composite::window_function::hamming_window);
  module.impl("periodic_hamming_window", &torchscience::composite::window_function::periodic_hamming_window);
  module.impl("blackman_window", &torchscience::composite::window_function::blackman_window);
  module.impl("periodic_blackman_window", &torchscience::composite::window_function::periodic_blackman_window);
  module.impl("bartlett_window", &torchscience::composite::window_function::bartlett_window);
  module.impl("periodic_bartlett_window", &torchscience::composite::window_function::periodic_bartlett_window);
  module.impl("cosine_window", &torchscience::composite::window_function::cosine_window);
  module.impl("periodic_cosine_window", &torchscience::composite::window_function::periodic_cosine_window);
  module.impl("nuttall_window", &torchscience::composite::window_function::nuttall_window);
  module.impl("periodic_nuttall_window", &torchscience::composite::window_function::periodic_nuttall_window);
  module.impl("triangular_window", &torchscience::composite::window_function::triangular_window);
  module.impl("periodic_triangular_window", &torchscience::composite::window_function::periodic_triangular_window);
  module.impl("welch_window", &torchscience::composite::window_function::welch_window);
  module.impl("periodic_welch_window", &torchscience::composite::window_function::periodic_welch_window);
  module.impl("parzen_window", &torchscience::composite::window_function::parzen_window);
  module.impl("periodic_parzen_window", &torchscience::composite::window_function::periodic_parzen_window);
  module.impl("blackman_harris_window", &torchscience::composite::window_function::blackman_harris_window);
  module.impl("periodic_blackman_harris_window", &torchscience::composite::window_function::periodic_blackman_harris_window);
  module.impl("flat_top_window", &torchscience::composite::window_function::flat_top_window);
  module.impl("periodic_flat_top_window", &torchscience::composite::window_function::periodic_flat_top_window);
  module.impl("sine_window", &torchscience::composite::window_function::sine_window);
  module.impl("periodic_sine_window", &torchscience::composite::window_function::periodic_sine_window);
  module.impl("bartlett_hann_window", &torchscience::composite::window_function::bartlett_hann_window);
  module.impl("periodic_bartlett_hann_window", &torchscience::composite::window_function::periodic_bartlett_hann_window);
  module.impl("lanczos_window", &torchscience::composite::window_function::lanczos_window);
  module.impl("periodic_lanczos_window", &torchscience::composite::window_function::periodic_lanczos_window);

  module.impl("gaussian_window", &torchscience::composite::window_function::gaussian_window);
  module.impl("periodic_gaussian_window", &torchscience::composite::window_function::periodic_gaussian_window);
  module.impl("general_hamming_window", &torchscience::composite::window_function::general_hamming_window);
  module.impl("periodic_general_hamming_window", &torchscience::composite::window_function::periodic_general_hamming_window);
  module.impl("general_cosine_window", &torchscience::composite::window_function::general_cosine_window);
  module.impl("periodic_general_cosine_window", &torchscience::composite::window_function::periodic_general_cosine_window);
  module.impl("tukey_window", &torchscience::composite::window_function::tukey_window);
  module.impl("periodic_tukey_window", &torchscience::composite::window_function::periodic_tukey_window);
}
