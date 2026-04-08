#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// Autocast Pointwise Macros - Modular
// complex parameter is accepted but ignored

#define TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_DISPATCH(category, name, arg1)                       \
namespace torchscience::autocast::category {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKeySet{c10::DispatchKey::AutocastCPU,                         \
                        c10::DispatchKey::AutocastCUDA}                        \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                     \
  auto target_type =                                                           \
    at::autocast::get_lower_precision_fp_from_device_type(device_type);        \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1, device_type)                \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::category */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {                       \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}                                                                              \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {                        \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_DISPATCH(category, name, arg1, arg2)                \
namespace torchscience::autocast::category {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKeySet{c10::DispatchKey::AutocastCPU,                         \
                        c10::DispatchKey::AutocastCUDA}                        \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                     \
  auto target_type =                                                           \
    at::autocast::get_lower_precision_fp_from_device_type(device_type);        \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1, device_type),               \
      at::autocast::cached_cast(target_type, arg2, device_type)                \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::category */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {                       \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}                                                                              \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {                        \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_DISPATCH(category, name, arg1, arg2, arg3)         \
namespace torchscience::autocast::category {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKeySet{c10::DispatchKey::AutocastCPU,                         \
                        c10::DispatchKey::AutocastCUDA}                        \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                     \
  auto target_type =                                                           \
    at::autocast::get_lower_precision_fp_from_device_type(device_type);        \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1, device_type),               \
      at::autocast::cached_cast(target_type, arg2, device_type),               \
      at::autocast::cached_cast(target_type, arg3, device_type)                \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::category */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {                       \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}                                                                              \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {                        \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_DISPATCH(category, name, arg1, arg2, arg3, arg4) \
namespace torchscience::autocast::category {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKeySet{c10::DispatchKey::AutocastCPU,                         \
                        c10::DispatchKey::AutocastCUDA}                        \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                     \
  auto target_type =                                                           \
    at::autocast::get_lower_precision_fp_from_device_type(device_type);        \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1, device_type),               \
      at::autocast::cached_cast(target_type, arg2, device_type),               \
      at::autocast::cached_cast(target_type, arg3, device_type),               \
      at::autocast::cached_cast(target_type, arg4, device_type)                \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::category */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {                       \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}                                                                              \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {                        \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUINARY_DISPATCH(category, name, arg1, arg2, arg3, arg4, arg5) \
namespace torchscience::autocast::category {                          \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &(arg1),                                                    \
  const at::Tensor &(arg2),                                                    \
  const at::Tensor &(arg3),                                                    \
  const at::Tensor &(arg4),                                                    \
  const at::Tensor &(arg5)                                                     \
) {                                                                            \
  c10::impl::ExcludeDispatchKeyGuard no_autocast(                              \
    c10::DispatchKeySet{c10::DispatchKey::AutocastCPU,                         \
                        c10::DispatchKey::AutocastCUDA}                        \
  );                                                                           \
                                                                               \
  auto device_type = arg1.device().type();                                     \
  auto target_type =                                                           \
    at::autocast::get_lower_precision_fp_from_device_type(device_type);        \
                                                                               \
  return c10::Dispatcher::singleton()                                          \
    .findSchemaOrThrow(                                                        \
      "torchscience::" #name,                                                  \
      ""                                                                       \
    ).typed<at::Tensor(                                                        \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>().call(                                                                 \
      at::autocast::cached_cast(target_type, arg1, device_type),               \
      at::autocast::cached_cast(target_type, arg2, device_type),               \
      at::autocast::cached_cast(target_type, arg3, device_type),               \
      at::autocast::cached_cast(target_type, arg4, device_type),               \
      at::autocast::cached_cast(target_type, arg5, device_type)                \
    );                                                                         \
}                                                                              \
                                                                               \
} /* namespace torchscience::autocast::category */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {                       \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}                                                                              \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {                        \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autocast::category::name                            \
  );                                                                           \
}

#define TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY(category, complex, name, arg1) \
    TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_DISPATCH(category, name, arg1)

#define TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY(category, complex, name, arg1, arg2) \
    TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_DISPATCH(category, name, arg1, arg2)

#define TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY(category, complex, name, arg1, arg2, arg3) \
    TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_DISPATCH(category, name, arg1, arg2, arg3)

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY(category, complex, name, arg1, arg2, arg3, arg4) \
    TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_DISPATCH(category, name, arg1, arg2, arg3, arg4)

#define TORCHSCIENCE_AUTOCAST_POINTWISE_QUINARY(category, complex, name, arg1, arg2, arg3, arg4, arg5) \
    TORCHSCIENCE_AUTOCAST_POINTWISE_QUINARY_DISPATCH(category, name, arg1, arg2, arg3, arg4, arg5)
