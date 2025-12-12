#pragma once

#include <torch/torch.h>

#define TORCHSCIENCE_UNARY_AUTOGRAD(NAME, SCHEMA_NAME)                          \
  class NAME                                                                    \
    : public torch::autograd::Function<NAME> {                                  \
  public:                                                                       \
    static torch::autograd::variable_list forward(                              \
      torch::autograd::AutogradContext* context,                                \
      const at::Tensor& input                                                   \
    ) {                                                                         \
      context->save_for_backward(                                               \
        {                                                                       \
          input                                                                 \
        }                                                                       \
      );                                                                        \
                                                                                \
      at::AutoDispatchBelowAutograd guard;                                      \
                                                                                \
      return {                                                                  \
        c10::Dispatcher::singleton().findSchemaOrThrow(                         \
          "torchscience::_" #SCHEMA_NAME,                                       \
          ""                                                                    \
        ).typed<at::Tensor(                                                     \
          const at::Tensor&                                                     \
        )>().call(                                                              \
          input                                                                 \
        )                                                                       \
      };                                                                        \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
      const torch::autograd::AutogradContext* context,                          \
      const torch::autograd::variable_list& gradient_output                     \
    ) {                                                                         \
      at::AutoDispatchBelowAutograd guard;                                      \
                                                                                \
      return {                                                                  \
        c10::Dispatcher::singleton().findSchemaOrThrow(                         \
          "torchscience::_" #SCHEMA_NAME "_backward",                           \
          ""                                                                    \
        ).typed<at::Tensor(                                                     \
          const at::Tensor&,                                                    \
          const at::Tensor&                                                     \
        )>().call(                                                              \
          gradient_output[0],                                                   \
          context->get_saved_variables()[0]                                     \
        )                                                                       \
      };                                                                        \
    }                                                                           \
  };                                                                            \
                                                                                \
  inline at::Tensor SCHEMA_NAME(                                                \
    const at::Tensor& input                                                     \
  ) {                                                                           \
    return NAME::apply(                                                         \
      input                                                                     \
    )[0];                                                                       \
  }

#define TORCHSCIENCE_UNARY_AUTOGRAD_IMPL(SCHEMA_NAME)                           \
  TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_BINARY_AUTOGRAD(NAME, SCHEMA_NAME, ARG0, ARG1)             \
  class NAME                                                                    \
    : public torch::autograd::Function<NAME> {                                  \
  public:                                                                       \
    static torch::autograd::variable_list forward(                              \
      torch::autograd::AutogradContext* context,                                \
      const at::Tensor& ARG0,                                                   \
      const at::Tensor& ARG1                                                    \
    ) {                                                                         \
      context->save_for_backward(                                               \
        {                                                                       \
          ARG0,                                                                 \
          ARG1                                                                  \
        }                                                                       \
      );                                                                        \
                                                                                \
      at::AutoDispatchBelowAutograd guard;                                      \
                                                                                \
      return {                                                                  \
        c10::Dispatcher::singleton().findSchemaOrThrow(                         \
          "torchscience::_" #SCHEMA_NAME,                                       \
          ""                                                                    \
        ).typed<at::Tensor(                                                     \
          const at::Tensor&,                                                    \
          const at::Tensor&                                                     \
        )>().call(                                                              \
          ARG0,                                                                 \
          ARG1                                                                  \
        )                                                                       \
      };                                                                        \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
      const torch::autograd::AutogradContext* context,                          \
      const torch::autograd::variable_list& gradient_output                     \
    ) {                                                                         \
      at::AutoDispatchBelowAutograd guard;                                      \
                                                                                \
      auto saved = context->get_saved_variables();                              \
                                                                                \
      auto grads = c10::Dispatcher::singleton().findSchemaOrThrow(              \
        "torchscience::_" #SCHEMA_NAME "_backward",                             \
        ""                                                                      \
      ).typed<std::tuple<at::Tensor, at::Tensor>(                               \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_output[0],                                                     \
        saved[0],                                                               \
        saved[1]                                                                \
      );                                                                        \
                                                                                \
      return {                                                                  \
        std::get<0>(grads),                                                     \
        std::get<1>(grads)                                                      \
      };                                                                        \
    }                                                                           \
  };                                                                            \
                                                                                \
  inline at::Tensor SCHEMA_NAME(                                                \
    const at::Tensor& ARG0,                                                     \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    return NAME::apply(                                                         \
      ARG0,                                                                     \
      ARG1                                                                      \
    )[0];                                                                       \
  }

#define TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(SCHEMA_NAME)                          \
  TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                          \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }
