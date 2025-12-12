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
