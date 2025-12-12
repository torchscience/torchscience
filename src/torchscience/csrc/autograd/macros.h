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
