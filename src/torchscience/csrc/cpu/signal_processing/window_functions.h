#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/signal_processing/window_function/hann.h"
#include "../../kernel/signal_processing/window_function/hamming.h"
#include "../../kernel/signal_processing/window_function/blackman.h"
#include "../../kernel/signal_processing/window_function/bartlett.h"
#include "../../kernel/signal_processing/window_function/cosine.h"
#include "../../kernel/signal_processing/window_function/nuttall.h"
#include "../../kernel/signal_processing/window_function/gaussian.h"
#include "../../kernel/signal_processing/window_function/general_hamming.h"
#include "../../kernel/signal_processing/window_function/general_cosine.h"
#include "../../kernel/signal_processing/window_function/triangular.h"
#include "../../kernel/signal_processing/window_function/welch.h"

namespace torchscience::cpu::window_function {

namespace {

inline at::TensorOptions build_window_options(
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCPU));
}

}  // anonymous namespace

// =============================================================================
// Parameterless windows macro
// =============================================================================

#define TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(name)                          \
inline at::Tensor name##_window_impl(                                           \
  int64_t n,                                                                    \
  bool periodic,                                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  TORCH_CHECK(n >= 0, #name "_window: n must be non-negative, got ", n);        \
                                                                                \
  auto options = build_window_options(dtype, layout, device);                   \
  auto output = at::empty({n}, options);                                        \
                                                                                \
  if (n == 0) {                                                                 \
    return output.requires_grad_(requires_grad);                                \
  }                                                                             \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    output.scalar_type(),                                                       \
    #name "_window",                                                            \
    [&] {                                                                       \
      auto* out_ptr = output.data_ptr<scalar_t>();                              \
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {            \
        for (int64_t i = begin; i < end; ++i) {                                 \
          out_ptr[i] = kernel::window_function::name<scalar_t>(i, n, periodic); \
        }                                                                       \
      });                                                                       \
    }                                                                           \
  );                                                                            \
                                                                                \
  return output.requires_grad_(requires_grad);                                  \
}                                                                               \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, false, dtype, layout, device, requires_grad);    \
}                                                                               \
                                                                                \
inline at::Tensor periodic_##name##_window(                                     \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, true, dtype, layout, device, requires_grad);     \
}

TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(hann)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(hamming)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(blackman)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(bartlett)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(cosine)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(nuttall)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(triangular)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(welch)

#undef TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW

// =============================================================================
// Gaussian window
// =============================================================================

inline at::Tensor gaussian_window_impl(
  int64_t n,
  const at::Tensor& std_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "gaussian_window: n must be non-negative, got ", n);
  TORCH_CHECK(std_input.dim() == 0, "gaussian_window: std must be a scalar tensor");

  auto out_dtype = dtype.value_or(std_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(std_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "gaussian_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t std_val = std_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::gaussian<scalar_t>(i, n, std_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window_impl(n, std_input, false, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window_impl(n, std_input, true, dtype, layout, device);
}

inline at::Tensor gaussian_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input,
  bool periodic
) {
  auto grad_std = at::zeros_like(std_input);

  if (n == 0) {
    return grad_std;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "gaussian_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t std_val = std_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::gaussian_backward<scalar_t>(
          grad_out_ptr[i], i, n, std_val, periodic, out_ptr[i]
        );
      }

      grad_std.fill_(accum);
    }
  );

  return grad_std;
}

inline at::Tensor gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward_impl(grad_output, output, n, std_input, false);
}

inline at::Tensor periodic_gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward_impl(grad_output, output, n, std_input, true);
}

// =============================================================================
// General Hamming window
// =============================================================================

inline at::Tensor general_hamming_window_impl(
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "general_hamming_window: n must be non-negative, got ", n);
  TORCH_CHECK(alpha_input.dim() == 0, "general_hamming_window: alpha must be a scalar tensor");

  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(alpha_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "general_hamming_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::general_hamming<scalar_t>(i, n, alpha_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window_impl(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window_impl(n, alpha_input, true, dtype, layout, device);
}

inline at::Tensor general_hamming_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic
) {
  auto grad_alpha = at::zeros_like(alpha_input);

  if (n == 0) {
    return grad_alpha;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "general_hamming_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::general_hamming_backward<scalar_t>(
          grad_out_ptr[i], i, n, alpha_val, periodic, out_ptr[i]
        );
      }

      grad_alpha.fill_(accum);
    }
  );

  return grad_alpha;
}

inline at::Tensor general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward_impl(grad_output, output, n, alpha_input, false);
}

inline at::Tensor periodic_general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward_impl(grad_output, output, n, alpha_input, true);
}

// =============================================================================
// General Cosine window
// =============================================================================

inline at::Tensor general_cosine_window_impl(
  int64_t n,
  const at::Tensor& coeffs_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "general_cosine_window: n must be non-negative, got ", n);
  TORCH_CHECK(coeffs_input.dim() == 1, "general_cosine_window: coeffs must be a 1-D tensor");
  TORCH_CHECK(coeffs_input.size(0) > 0, "general_cosine_window: coeffs must have at least one element");

  auto out_dtype = dtype.value_or(coeffs_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(coeffs_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  int64_t num_coeffs = coeffs_input.size(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "general_cosine_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      auto coeffs_contig = coeffs_input.to(output.scalar_type()).contiguous();
      auto* coeffs_ptr = coeffs_contig.data_ptr<scalar_t>();

      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::general_cosine<scalar_t>(
            i, n, coeffs_ptr, num_coeffs, periodic
          );
        }
      });
    }
  );

  return output;
}

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, true, dtype, layout, device);
}

inline at::Tensor general_cosine_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input,
  bool periodic
) {
  (void)output;
  int64_t num_coeffs = coeffs_input.size(0);
  auto grad_coeffs = at::zeros_like(coeffs_input);

  if (n == 0) {
    return grad_coeffs;
  }

  // Ensure contiguous tensor for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "general_cosine_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto grad_coeffs_accessor = grad_coeffs.accessor<scalar_t, 1>();

      for (int64_t j = 0; j < num_coeffs; ++j) {
        scalar_t accum = scalar_t(0);
        for (int64_t i = 0; i < n; ++i) {
          accum += kernel::window_function::general_cosine_backward_coeff<scalar_t>(
            grad_out_ptr[i], i, n, j, periodic
          );
        }
        grad_coeffs_accessor[j] = accum;
      }
    }
  );

  return grad_coeffs;
}

inline at::Tensor general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward_impl(grad_output, output, n, coeffs_input, false);
}

inline at::Tensor periodic_general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward_impl(grad_output, output, n, coeffs_input, true);
}

}  // namespace torchscience::cpu::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("hann_window", torchscience::cpu::window_function::hann_window);
  m.impl("periodic_hann_window", torchscience::cpu::window_function::periodic_hann_window);
  m.impl("hamming_window", torchscience::cpu::window_function::hamming_window);
  m.impl("periodic_hamming_window", torchscience::cpu::window_function::periodic_hamming_window);
  m.impl("blackman_window", torchscience::cpu::window_function::blackman_window);
  m.impl("periodic_blackman_window", torchscience::cpu::window_function::periodic_blackman_window);
  m.impl("bartlett_window", torchscience::cpu::window_function::bartlett_window);
  m.impl("periodic_bartlett_window", torchscience::cpu::window_function::periodic_bartlett_window);
  m.impl("cosine_window", torchscience::cpu::window_function::cosine_window);
  m.impl("periodic_cosine_window", torchscience::cpu::window_function::periodic_cosine_window);
  m.impl("nuttall_window", torchscience::cpu::window_function::nuttall_window);
  m.impl("periodic_nuttall_window", torchscience::cpu::window_function::periodic_nuttall_window);
  m.impl("triangular_window", torchscience::cpu::window_function::triangular_window);
  m.impl("periodic_triangular_window", torchscience::cpu::window_function::periodic_triangular_window);
  m.impl("welch_window", torchscience::cpu::window_function::welch_window);
  m.impl("periodic_welch_window", torchscience::cpu::window_function::periodic_welch_window);

  m.impl("gaussian_window", torchscience::cpu::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::cpu::window_function::periodic_gaussian_window);
  m.impl("gaussian_window_backward", torchscience::cpu::window_function::gaussian_window_backward);
  m.impl("periodic_gaussian_window_backward", torchscience::cpu::window_function::periodic_gaussian_window_backward);

  m.impl("general_hamming_window", torchscience::cpu::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::cpu::window_function::periodic_general_hamming_window);
  m.impl("general_hamming_window_backward", torchscience::cpu::window_function::general_hamming_window_backward);
  m.impl("periodic_general_hamming_window_backward", torchscience::cpu::window_function::periodic_general_hamming_window_backward);

  m.impl("general_cosine_window", torchscience::cpu::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::cpu::window_function::periodic_general_cosine_window);
  m.impl("general_cosine_window_backward", torchscience::cpu::window_function::general_cosine_window_backward);
  m.impl("periodic_general_cosine_window_backward", torchscience::cpu::window_function::periodic_general_cosine_window_backward);
}
