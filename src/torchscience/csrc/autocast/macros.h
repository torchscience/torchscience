#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

#define TORCHSCIENCE_UNARY_AUTOCAST(SCHEMA_NAME)                                \
  at::Tensor SCHEMA_NAME(const at::Tensor& input) {                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
      c10::DispatchKey::Autocast                                                \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          input                                                                 \
        )                                                                       \
      );                                                                        \
  }

#define TORCHSCIENCE_UNARY_AUTOCAST_IMPL(SCHEMA_NAME)                           \
  TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_AUTOCAST(SCHEMA_NAME, ARG0, ARG1)                   \
  at::Tensor SCHEMA_NAME(                                                       \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
      c10::DispatchKey::Autocast                                                \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG0                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG1                                                                  \
        )                                                                       \
      );                                                                        \
  }

#define TORCHSCIENCE_BINARY_AUTOCAST_IMPL(SCHEMA_NAME)                          \
  TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_TERNARY_AUTOCAST(SCHEMA_NAME, ARG0, ARG1, ARG2)            \
  at::Tensor SCHEMA_NAME(                                                       \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
      c10::DispatchKey::Autocast                                                \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG0                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG1                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG2                                                                  \
        )                                                                       \
      );                                                                        \
  }

#define TORCHSCIENCE_TERNARY_AUTOCAST_IMPL(SCHEMA_NAME)                         \
  TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_QUATERNARY_AUTOCAST(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3)   \
  at::Tensor SCHEMA_NAME(                                                       \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
      c10::DispatchKey::Autocast                                                \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG0                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG1                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG2                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG3                                                                  \
        )                                                                       \
      );                                                                        \
  }

#define TORCHSCIENCE_QUATERNARY_AUTOCAST_IMPL(SCHEMA_NAME)                      \
  TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_QUINARY_AUTOCAST(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3, ARG4) \
  at::Tensor SCHEMA_NAME(                                                       \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const at::Tensor& ARG4                                                      \
  ) {                                                                           \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
      c10::DispatchKey::Autocast                                                \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::_" #SCHEMA_NAME,                                         \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG0                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG1                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG2                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG3                                                                  \
        ),                                                                      \
        at::autocast::cached_cast(                                              \
          at::kFloat,                                                           \
          ARG4                                                                  \
        )                                                                       \
      );                                                                        \
  }

#define TORCHSCIENCE_QUINARY_AUTOCAST_IMPL(SCHEMA_NAME)                         \
  TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }
