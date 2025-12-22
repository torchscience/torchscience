#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// =============================================================================
// Helper macros for handling parenthesized parameter lists
// =============================================================================

#ifndef TORCHSCIENCE_UNPACK_IMPL
#define TORCHSCIENCE_UNPACK_IMPL(...) __VA_ARGS__
#define TORCHSCIENCE_UNPACK(X) TORCHSCIENCE_UNPACK_IMPL X
#define TORCHSCIENCE_COMMA_IF_IMPL(...) __VA_OPT__(,)
#define TORCHSCIENCE_COMMA_IF(X) TORCHSCIENCE_COMMA_IF_IMPL(TORCHSCIENCE_UNPACK(X))
#endif

// =============================================================================
// AUTOCAST_CREATION_OPERATOR
// =============================================================================
// Flexible macro for creation operators under autocast.
// When autocast is enabled and no dtype is specified, uses autocast dtype.
//
// Parameters:
//   NAMESPACE     - Namespace (e.g., window_function)
//   OPERATOR_NAME - Operator name (e.g., rectangular_window)
//   PARAMS        - Parenthesized typed params: (int64_t n) or ()
//   ARGS          - Parenthesized arg names: (n) or ()
//   TYPED_SIG     - Typed signature for dispatcher (params + standard opts)
//
// Usage:
//   AUTOCAST_CREATION_OPERATOR(
//     window_function, rectangular_window,
//     (int64_t n), (n),
//     (int64_t, const c10::optional<at::ScalarType>&,
//      const c10::optional<at::Layout>&, const c10::optional<at::Device>&, bool))
// =============================================================================

#define AUTOCAST_CREATION_OPERATOR(                                             \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  PARAMS,                                                                       \
  ARGS,                                                                         \
  TYPED_SIG                                                                     \
)                                                                               \
                                                                                \
namespace torchscience::autocast::NAMESPACE {                                   \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  TORCHSCIENCE_UNPACK(PARAMS)                                                   \
  TORCHSCIENCE_COMMA_IF(PARAMS)                                                 \
  const c10::optional<at::ScalarType>& dtype,                                   \
  const c10::optional<at::Layout>& layout,                                      \
  const c10::optional<at::Device>& device,                                      \
  bool requires_grad                                                            \
) {                                                                             \
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast(                          \
    c10::DispatchKey::Autocast                                                  \
  );                                                                            \
                                                                                \
  /* If dtype not specified, use autocast dtype based on device */              \
  c10::optional<at::ScalarType> effective_dtype = dtype;                        \
  if (!dtype.has_value()) {                                                     \
    auto dev = device.value_or(at::kCPU);                                       \
    if (dev.is_cuda()) {                                                        \
      effective_dtype = at::autocast::get_autocast_dtype(at::kCUDA);            \
    } else {                                                                    \
      effective_dtype = at::autocast::get_autocast_dtype(at::kCPU);             \
    }                                                                           \
  }                                                                             \
                                                                                \
  return c10::Dispatcher::singleton().findSchemaOrThrow(                        \
    "torchscience::" #OPERATOR_NAME,                                            \
    ""                                                                          \
  ).typed<at::Tensor TORCHSCIENCE_UNPACK(TYPED_SIG)>()                          \
  .call(                                                                        \
    TORCHSCIENCE_UNPACK(ARGS)                                                   \
    TORCHSCIENCE_COMMA_IF(ARGS)                                                 \
    effective_dtype,                                                            \
    layout,                                                                     \
    device,                                                                     \
    requires_grad                                                               \
  );                                                                            \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NAMESPACE */                             \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                            \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::autocast::NAMESPACE::OPERATOR_NAME                           \
  );                                                                            \
}
