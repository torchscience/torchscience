#pragma once

#include <torch/extension.h>

#define TORCHSCIENCE_AUTOGRAD_UNARY_OPERATOR(name, Name, arg)                  \
namespace torchscience::autograd::special_functions {                          \
                                                                               \
class Name##Backward : public torch::autograd::Function<Name##Backward> {      \
public:                                                                        \
  static std::vector<at::Tensor> forward(                                      \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &gradient_output,                                         \
    const at::Tensor &input,                                                   \
    bool input_requires_gradient                                               \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        gradient_output,                                                       \
        input                                                                  \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data["input_requires_grad"] = input_requires_gradient;      \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    static auto op = c10::Dispatcher::singleton()                              \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward",                                    \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &,                                                    \
        const at::Tensor &                                                     \
      )>();                                                                    \
                                                                               \
    return {                                                                   \
      op.call(                                                                 \
        gradient_output,                                                       \
        input                                                                  \
      )                                                                        \
    };                                                                         \
  }                                                                            \
                                                                               \
  static std::vector<at::Tensor> backward(                                     \
    torch::autograd::AutogradContext *context,                                 \
    const std::vector<at::Tensor>                                              \
    &gradient_outputs                                                          \
  ) {                                                                          \
    torch::autograd::variable_list saved_variables = context->get_saved_variables(); \
                                                                               \
    bool input_requires_grad = context->saved_data["input_requires_grad"].toBool(); \
                                                                               \
    if (!gradient_outputs[0].defined() || !input_requires_grad) {              \
      return {                                                                 \
        at::Tensor(),                                                          \
        at::Tensor(),                                                          \
        at::Tensor()                                                           \
      };                                                                       \
    }                                                                          \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    auto [                                                                     \
      gradient_gradient_output,                                                \
      gradient_output                                                          \
    ] = c10::Dispatcher::singleton()                                           \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name "_backward_backward",                           \
        ""                                                                     \
      )                                                                        \
    .typed<std::tuple<                                                         \
      at::Tensor,                                                              \
      at::Tensor                                                               \
    >(                                                                         \
      const at::Tensor &,                                                      \
      const at::Tensor &,                                                      \
      const at::Tensor &                                                       \
    )>()                                                                       \
    .call(                                                                     \
      gradient_outputs[0],                                                     \
      saved_variables[0],                                                      \
      saved_variables[1]                                                       \
    );                                                                         \
                                                                               \
    return {                                                                   \
      gradient_gradient_output,                                                \
      gradient_output,                                                         \
      at::Tensor()                                                             \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
class Name : public torch::autograd::Function<Name> {                          \
public:                                                                        \
  static at::Tensor forward(                                                   \
    torch::autograd::AutogradContext *context,                                 \
    const at::Tensor &input                                                    \
  ) {                                                                          \
    context->save_for_backward(                                                \
      {                                                                        \
        input                                                                  \
      }                                                                        \
    );                                                                         \
                                                                               \
    context->saved_data["input_requires_grad"] = input.requires_grad() && (isFloatingType(input.scalar_type()) || isComplexType(input.scalar_type())); \
                                                                               \
    at::AutoDispatchBelowAutograd guard;                                       \
                                                                               \
    return c10::Dispatcher::singleton()                                        \
      .findSchemaOrThrow(                                                      \
        "torchscience::" #name,                                                \
        ""                                                                     \
      )                                                                        \
      .typed<at::Tensor(                                                       \
        const at::Tensor &                                                     \
      )>()                                                                     \
      .call(                                                                   \
        input                                                                  \
      );                                                                       \
  }                                                                            \
                                                                               \
  static torch::autograd::variable_list backward(                              \
    torch::autograd::AutogradContext *context,                                 \
    const torch::autograd::variable_list &gradient_outputs                     \
  ) {                                                                          \
    torch::autograd::variable_list variables = context->get_saved_variables(); \
                                                                               \
    bool input_requires_grad = context->saved_data["input_requires_grad"].toBool(); \
                                                                               \
    auto gradients = Name##Backward::apply(                                    \
      gradient_outputs[0],                                                     \
      variables[0],                                                            \
      input_requires_grad                                                      \
    );                                                                         \
                                                                               \
    at::Tensor output;                                                         \
                                                                               \
    if (input_requires_grad) {                                                 \
      output = gradients[0];                                                   \
    } else {                                                                   \
      output = at::Tensor();                                                   \
    }                                                                          \
                                                                               \
    return {                                                                   \
      output                                                                   \
    };                                                                         \
  }                                                                            \
};                                                                             \
                                                                               \
inline at::Tensor name(                                                        \
  const at::Tensor &arg                                                        \
) {                                                                            \
  return Name::apply(arg);                                                     \
}                                                                              \
                                                                               \
} /* namespace torchscience::autograd::special_functions */                    \
                                                                               \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                           \
  module.impl(                                                                 \
    #name,                                                                     \
    torchscience::autograd::special_functions::name                            \
  );                                                                           \
}
