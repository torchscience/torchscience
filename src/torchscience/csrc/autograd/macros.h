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

#define TORCHSCIENCE_BINARY_AUTOGRAD(NAME, SCHEMA_NAME, ARG1, ARG2)             \
  class NAME                                                                    \
    : public torch::autograd::Function<NAME> {                                  \
  public:                                                                       \
    static torch::autograd::variable_list forward(                              \
      torch::autograd::AutogradContext* context,                                \
      const at::Tensor& ARG1,                                                   \
      const at::Tensor& ARG2                                                    \
    ) {                                                                         \
      context->save_for_backward(                                               \
        {                                                                       \
          ARG1,                                                                 \
          ARG2                                                                  \
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
          ARG1,                                                                 \
          ARG2                                                                  \
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
      auto ARG1 = saved[0];                                                     \
      auto ARG2 = saved[1];                                                     \
                                                                                \
      auto gradients = c10::Dispatcher::singleton().findSchemaOrThrow(          \
        "torchscience::_" #SCHEMA_NAME "_backward",                             \
        ""                                                                      \
      ).typed<at::Tensor(                                                       \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_output[0],                                                     \
        ARG1,                                                                   \
        ARG2                                                                    \
      );                                                                        \
                                                                                \
      return {                                                                  \
        gradients[0],                                                           \
        gradients[1]                                                            \
      };                                                                        \
    }                                                                           \
  };                                                                            \
                                                                                \
  inline at::Tensor SCHEMA_NAME(                                                \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    return NAME::apply(                                                         \
      ARG1,                                                                     \
      ARG2                                                                      \
    )[0];                                                                       \
  }

#define TORCHSCIENCE_BINARY_AUTOGRAD_IMPL(SCHEMA_NAME)                          \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    Autograd,                                                                   \
    module                                                                      \
  ) {                                                                           \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }

#define TORCHSCIENCE_TERNARY_AUTOGRAD(NAME, SCHEMA_NAME, ARG1, ARG2, ARG3)      \
  class NAME                                                                    \
    : public torch::autograd::Function<NAME> {                                  \
  public:                                                                       \
    static torch::autograd::variable_list forward(                              \
      torch::autograd::AutogradContext* context,                                \
      const at::Tensor& ARG1,                                                   \
      const at::Tensor& ARG2,                                                   \
      const at::Tensor& ARG3                                                    \
    ) {                                                                         \
      context->save_for_backward(                                               \
        {                                                                       \
          ARG1,                                                                 \
          ARG2,                                                                 \
          ARG3                                                                  \
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
          const at::Tensor&,                                                    \
          const at::Tensor&                                                     \
        )>().call(                                                              \
          ARG1,                                                                 \
          ARG2,                                                                 \
          ARG3                                                                  \
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
      auto ARG1 = saved[0];                                                     \
      auto ARG2 = saved[1];                                                     \
      auto ARG3 = saved[2];                                                     \
                                                                                \
      auto gradients = c10::Dispatcher::singleton().findSchemaOrThrow(          \
        "torchscience::_" #SCHEMA_NAME "_backward",                             \
        ""                                                                      \
      ).typed<at::Tensor(                                                       \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_output[0],                                                     \
        ARG1,                                                                   \
        ARG2,                                                                   \
        ARG3                                                                    \
      );                                                                        \
                                                                                \
      return {                                                                  \
        gradients[0],                                                           \
        gradients[1],                                                           \
        gradients[2]                                                            \
      };                                                                        \
    }                                                                           \
  };                                                                            \
                                                                                \
  inline at::Tensor SCHEMA_NAME(                                                \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    return NAME::apply(                                                         \
      ARG1,                                                                     \
      ARG2,                                                                     \
      ARG3                                                                      \
    )[0];                                                                       \
  }

#define TORCHSCIENCE_TERNARY_AUTOGRAD_IMPL(SCHEMA_NAME)                         \
  TORCH_LIBRARY_IMPL(                                                           \
    torchscience,                                                               \
    Autograd,                                                                   \
    module                                                                      \
  ) {                                                                           \
    module.impl(                                                                \
      "_" #SCHEMA_NAME,                                                         \
      &SCHEMA_NAME                                                              \
    );                                                                          \
  }
