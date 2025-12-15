#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

#define AUTOCAST_UNARY_OPERATOR(                                                \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autocast::NAMESPACE {                                   \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast(                          \
    c10::DispatchKey::Autocast                                                  \
  );                                                                            \
                                                                                \
  at::ScalarType dtype;                                                         \
                                                                                \
  if (ARG1.device().is_cpu()) {                                                 \
    dtype = at::autocast::get_autocast_dtype(at::kCPU);                         \
  } else {                                                                      \
    dtype = at::autocast::get_autocast_dtype(at::kCUDA);                        \
  }                                                                             \
                                                                                \
  at::ScalarType target_dtype = dtype;                                          \
                                                                                \
  if (isComplexType(ARG1.scalar_type())) {                                      \
    if (dtype == at::kHalf) {                                                   \
      target_dtype = at::kComplexHalf;                                          \
    } else if (dtype == at::kBFloat16) {                                        \
      target_dtype = at::kComplexFloat;                                         \
    } else {                                                                    \
      target_dtype = at::kComplexFloat;                                         \
    }                                                                           \
  }                                                                             \
                                                                                \
  return c10::Dispatcher::singleton().findSchemaOrThrow(                        \
    "torchscience::" #OPERATOR_NAME,                                            \
    ""                                                                          \
  ).typed<at::Tensor(                                                           \
    const at::Tensor&                                                           \
  )>()                                                                          \
  .call(                                                                        \
    at::autocast::cached_cast(target_dtype, ARG1)                               \
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

#define AUTOCAST_BINARY_OPERATOR(                                               \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autocast::NAMESPACE {                                   \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast(                          \
    c10::DispatchKey::Autocast                                                  \
  );                                                                            \
                                                                                \
  at::ScalarType dtype;                                                         \
                                                                                \
  if (ARG1.device().is_cpu() && ARG2.device().is_cpu()) {                       \
    dtype = at::autocast::get_autocast_dtype(at::kCPU);                         \
  } else {                                                                      \
    dtype = at::autocast::get_autocast_dtype(at::kCUDA);                        \
  }                                                                             \
                                                                                \
  at::ScalarType target_dtype = dtype;                                          \
                                                                                \
  if (isComplexType(ARG1.scalar_type()) || isComplexType(ARG2.scalar_type())) { \
    if (dtype == at::kHalf) {                                                   \
      target_dtype = at::kComplexHalf;                                          \
    } else if (dtype == at::kBFloat16) {                                        \
      target_dtype = at::kComplexFloat;                                         \
    } else {                                                                    \
      target_dtype = at::kComplexFloat;                                         \
    }                                                                           \
  }                                                                             \
                                                                                \
  return c10::Dispatcher::singleton().findSchemaOrThrow(                        \
    "torchscience::" #OPERATOR_NAME,                                            \
    ""                                                                          \
  ).typed<at::Tensor(                                                           \
    const at::Tensor&,                                                          \
    const at::Tensor&                                                           \
  )>()                                                                          \
  .call(                                                                        \
    at::autocast::cached_cast(target_dtype, ARG1),                              \
    at::autocast::cached_cast(target_dtype, ARG2)                               \
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

#define AUTOCAST_TERNARY_OPERATOR(                                              \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autocast::NAMESPACE {                                   \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast(                          \
    c10::DispatchKey::Autocast                                                  \
  );                                                                            \
                                                                                \
  at::ScalarType dtype;                                                         \
                                                                                \
  if (ARG1.device().is_cpu() &&                                                 \
      ARG2.device().is_cpu() &&                                                 \
      ARG3.device().is_cpu()) {                                                 \
    dtype = at::autocast::get_autocast_dtype(at::kCPU);                         \
  } else {                                                                      \
    dtype = at::autocast::get_autocast_dtype(at::kCUDA);                        \
  }                                                                             \
                                                                                \
  at::ScalarType target_dtype = dtype;                                          \
                                                                                \
  if (isComplexType(ARG1.scalar_type()) ||                                      \
      isComplexType(ARG2.scalar_type()) ||                                      \
      isComplexType(ARG3.scalar_type())) {                                      \
    if (dtype == at::kHalf) {                                                   \
      target_dtype = at::kComplexHalf;                                          \
    } else if (dtype == at::kBFloat16) {                                        \
      target_dtype = at::kComplexFloat;                                         \
    } else {                                                                    \
      target_dtype = at::kComplexFloat;                                         \
    }                                                                           \
  }                                                                             \
                                                                                \
  return c10::Dispatcher::singleton().findSchemaOrThrow(                        \
    "torchscience::" #OPERATOR_NAME,                                            \
    ""                                                                          \
  ).typed<at::Tensor(                                                           \
    const at::Tensor&,                                                          \
    const at::Tensor&,                                                          \
    const at::Tensor&                                                           \
  )>()                                                                          \
  .call(                                                                        \
    at::autocast::cached_cast(target_dtype, ARG1),                              \
    at::autocast::cached_cast(target_dtype, ARG2),                              \
    at::autocast::cached_cast(target_dtype, ARG3)                               \
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

#define AUTOCAST_QUATERNARY_OPERATOR(                                           \
  NAMESPACE,                                                                    \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autocast::NAMESPACE {                                   \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast(                          \
    c10::DispatchKey::Autocast                                                  \
  );                                                                            \
                                                                                \
  at::ScalarType dtype;                                                         \
                                                                                \
  if (ARG1.device().is_cpu() &&                                                 \
      ARG2.device().is_cpu() &&                                                 \
      ARG3.device().is_cpu() &&                                                 \
      ARG4.device().is_cpu()) {                                                 \
    dtype = at::autocast::get_autocast_dtype(at::kCPU);                         \
  } else {                                                                      \
    dtype = at::autocast::get_autocast_dtype(at::kCUDA);                        \
  }                                                                             \
                                                                                \
  at::ScalarType target_dtype = dtype;                                          \
                                                                                \
  if (isComplexType(ARG1.scalar_type()) ||                                      \
      isComplexType(ARG2.scalar_type()) ||                                      \
      isComplexType(ARG3.scalar_type()) ||                                      \
      isComplexType(ARG4.scalar_type())) {                                      \
    if (dtype == at::kHalf) {                                                   \
      target_dtype = at::kComplexHalf;                                          \
    } else if (dtype == at::kBFloat16) {                                        \
      target_dtype = at::kComplexFloat;                                         \
    } else {                                                                    \
      target_dtype = at::kComplexFloat;                                         \
    }                                                                           \
  }                                                                             \
                                                                                \
  return c10::Dispatcher::singleton().findSchemaOrThrow(                        \
    "torchscience::" #OPERATOR_NAME,                                            \
    ""                                                                          \
  ).typed<at::Tensor(                                                           \
    const at::Tensor&,                                                          \
    const at::Tensor&,                                                          \
    const at::Tensor&,                                                          \
    const at::Tensor&                                                           \
  )>()                                                                          \
  .call(                                                                        \
    at::autocast::cached_cast(target_dtype, ARG1),                              \
    at::autocast::cached_cast(target_dtype, ARG2),                              \
    at::autocast::cached_cast(target_dtype, ARG3),                              \
    at::autocast::cached_cast(target_dtype, ARG4)                               \
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
