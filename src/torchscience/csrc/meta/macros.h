#pragma once

#include <torch/torch.h>

#define TORCHSCIENCE_UNARY_META_KERNEL(SCHEMA_NAME)                             \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& input                                                     \
  ) {                                                                           \
    return at::empty_like(                                                      \
      input,                                                                    \
      input.options().device(at::kMeta)                                         \
    );                                                                          \
  }                                                                             \
                                                                                \
  at::Tensor SCHEMA_NAME##_backward(                                            \
    [[maybe_unused]] const at::Tensor& gradient_output,                         \
    const at::Tensor& input                                                     \
  ) {                                                                           \
    return at::empty_like(                                                      \
      input,                                                                    \
      input.options().device(at::kMeta)                                         \
    );                                                                          \
  }

#define TORCHSCIENCE_UNARY_META_KERNEL_IMPL(SCHEMA_NAME)                        \
  TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                              \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward                                                    \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward                                                   \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_META_KERNEL(SCHEMA_NAME, ARG0, ARG1)                \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    return at::empty_like(                                                      \
      ARG0,                                                                     \
      ARG0.options().device(at::kMeta)                                          \
    );                                                                          \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor> SCHEMA_NAME##_backward(                    \
    [[maybe_unused]] const at::Tensor& gradient_output,                         \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    return std::make_tuple(                                                     \
      at::empty_like(                                                           \
        ARG0,                                                                   \
        ARG0.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG1,                                                                   \
        ARG1.options().device(at::kMeta)                                        \
      )                                                                         \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_META_KERNEL_IMPL(SCHEMA_NAME)                       \
  TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                              \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward                                                    \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward                                                   \
    );                                                                          \
  }

#define TORCHSCIENCE_TERNARY_META_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2)         \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    [[maybe_unused]] const at::Tensor& ARG1,                                    \
    [[maybe_unused]] const at::Tensor& ARG2                                     \
  ) {                                                                           \
    return at::empty_like(                                                      \
      ARG0,                                                                     \
      ARG0.options().device(at::kMeta)                                          \
    );                                                                          \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor, at::Tensor> SCHEMA_NAME##_backward(        \
    [[maybe_unused]] const at::Tensor& gradient_output,                         \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    return std::make_tuple(                                                     \
      at::empty_like(                                                           \
        ARG0,                                                                   \
        ARG0.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG1,                                                                   \
        ARG1.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG2,                                                                   \
        ARG2.options().device(at::kMeta)                                        \
      )                                                                         \
    );                                                                          \
  }

#define TORCHSCIENCE_TERNARY_META_KERNEL_IMPL(SCHEMA_NAME)                      \
  TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                              \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward                                                    \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward                                                   \
    );                                                                          \
  }

#define TORCHSCIENCE_QUATERNARY_META_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3) \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    [[maybe_unused]] const at::Tensor& ARG1,                                    \
    [[maybe_unused]] const at::Tensor& ARG2,                                    \
    [[maybe_unused]] const at::Tensor& ARG3                                     \
  ) {                                                                           \
    return at::empty_like(                                                      \
      ARG0,                                                                     \
      ARG0.options().device(at::kMeta)                                          \
    );                                                                          \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>                    \
  SCHEMA_NAME##_backward(                                                       \
    [[maybe_unused]] const at::Tensor& gradient_output,                         \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    return std::make_tuple(                                                     \
      at::empty_like(                                                           \
        ARG0,                                                                   \
        ARG0.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG1,                                                                   \
        ARG1.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG2,                                                                   \
        ARG2.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG3,                                                                   \
        ARG3.options().device(at::kMeta)                                        \
      )                                                                         \
    );                                                                          \
  }

#define TORCHSCIENCE_QUATERNARY_META_KERNEL_IMPL(SCHEMA_NAME)                   \
  TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                              \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward                                                    \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward                                                   \
    );                                                                          \
  }

#define TORCHSCIENCE_QUINARY_META_KERNEL(SCHEMA_NAME, ARG0, ARG1, ARG2, ARG3, ARG4) \
  at::Tensor SCHEMA_NAME##_forward(                                             \
    const at::Tensor& ARG0,                                                     \
    [[maybe_unused]] const at::Tensor& ARG1,                                    \
    [[maybe_unused]] const at::Tensor& ARG2,                                    \
    [[maybe_unused]] const at::Tensor& ARG3,                                    \
    [[maybe_unused]] const at::Tensor& ARG4                                     \
  ) {                                                                           \
    return at::empty_like(                                                      \
      ARG0,                                                                     \
      ARG0.options().device(at::kMeta)                                          \
    );                                                                          \
  }                                                                             \
                                                                                \
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>        \
  SCHEMA_NAME##_backward(                                                       \
    [[maybe_unused]] const at::Tensor& gradient_output,                         \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const at::Tensor& ARG4                                                      \
  ) {                                                                           \
    return std::make_tuple(                                                     \
      at::empty_like(                                                           \
        ARG0,                                                                   \
        ARG0.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG1,                                                                   \
        ARG1.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG2,                                                                   \
        ARG2.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG3,                                                                   \
        ARG3.options().device(at::kMeta)                                        \
      ),                                                                        \
      at::empty_like(                                                           \
        ARG4,                                                                   \
        ARG4.options().device(at::kMeta)                                        \
      )                                                                         \
    );                                                                          \
  }

#define TORCHSCIENCE_QUINARY_META_KERNEL_IMPL(SCHEMA_NAME)                      \
  TORCH_LIBRARY_IMPL(torchscience, Meta, module) {                              \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME##_forward                                                    \
    );                                                                          \
                                                                                \
    module.impl(                                                                \
      "_" #SCHEMA_NAME "_backward",                                             \
      &SCHEMA_NAME##_backward                                                   \
    );                                                                          \
  }
