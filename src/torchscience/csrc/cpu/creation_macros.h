#pragma once

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

// =============================================================================
// Helper macros for handling parenthesized parameter lists
// =============================================================================

// Remove parentheses: (int64_t n, double f) -> int64_t n, double f
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X

// Add comma if non-empty: (n, f) -> ,  |  () -> (empty)
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))

// =============================================================================
// CPU_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creating tensors from scalar parameters.
//
// Parameters:
//   NAMESPACE     - Namespace (e.g., window_function)
//   OPERATOR_NAME - Operator name (e.g., rectangular_window)
//   OUTPUT_SHAPE  - Shape expression (e.g., {n})
//   PARAMS        - Parenthesized typed params: (int64_t n) or ()
//   ARGS          - Parenthesized arg names: (n) or ()
//
// Usage:
//   CPU_CREATION_OPERATOR(window_function, rectangular_window, {n}, (int64_t n), (n))
//   CPU_CREATION_OPERATOR(waveform, sine_wave, {n},
//     (int64_t n, double freq, double sr, double amp, double phase),
//     (n, freq, sr, amp, phase))
//
// Kernel signature expected:
//   namespace impl::NAMESPACE {
//     template <typename scalar_t>
//     void OPERATOR_NAME_kernel(scalar_t* output, int64_t numel, PARAMS...)
//   }
// =============================================================================

#define CPU_CREATION_OPERATOR(                                                  \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  OUTPUT_SHAPE,                                                                 \
  PARAMS,                                                                       \
  ARGS                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::cpu::NAMESPACE {                                        \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  auto options = at::TensorOptions()                                            \
    .dtype(dtype.value_or(                                                      \
      c10::typeMetaToScalarType(at::get_default_dtype())                        \
    ))                                                                          \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(device.value_or(at::kCPU))                                          \
    .requires_grad(false);                                                      \
                                                                                \
  std::vector<int64_t> shape_vec = OUTPUT_SHAPE;                                \
  for (auto s : shape_vec) {                                                    \
    TORCH_CHECK(s >= 0,                                                         \
      #OPERATOR_NAME ": size must be non-negative, got ", s);                   \
  }                                                                             \
                                                                                \
  int64_t numel = 1;                                                            \
  for (auto s : shape_vec) {                                                    \
    numel *= s;                                                                 \
  }                                                                             \
                                                                                \
  at::Tensor output = at::empty(shape_vec, options);                            \
                                                                                \
  if (numel > 0) {                                                              \
    AT_DISPATCH_FLOATING_TYPES_AND2(                                            \
      at::kBFloat16,                                                            \
      at::kHalf,                                                                \
      output.scalar_type(),                                                     \
      #OPERATOR_NAME,                                                           \
      [&]() {                                                                   \
        impl::NAMESPACE::OPERATOR_NAME##_kernel<scalar_t>(                      \
          output.data_ptr<scalar_t>(),                                          \
          numel                                                                 \
          TORCHSCIENCE_COMMA_IF(ARGS)                                           \
          TORCHSCIENCE_UNPACK(ARGS)                                             \
        );                                                                      \
      }                                                                         \
    );                                                                          \
  }                                                                             \
                                                                                \
  if (requires_grad) {                                                          \
    output = output.requires_grad_(true);                                       \
  }                                                                             \
                                                                                \
  return output;                                                                \
}                                                                               \
                                                                                \
}  /* namespace torchscience::cpu::NAMESPACE */                                 \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                                 \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::cpu::NAMESPACE::OPERATOR_NAME                                \
  );                                                                            \
}
