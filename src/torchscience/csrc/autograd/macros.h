#pragma once

#include <tuple>

#include <torch/extension.h>

#define AUTOGRAD_UNARY_OPERATOR(                                                \
  NAMESPACE,                                                                    \
  CLASS_NAME,                                                                   \
  OPERATOR_NAME,                                                                \
  ARG1                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autograd::NAMESPACE {                                   \
                                                                                \
class CLASS_NAME##Backward                                                      \
  : public torch::autograd::Function<CLASS_NAME##Backward> {                    \
public:                                                                         \
  static std::vector<at::Tensor> forward(                                       \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG1,                                                     \
    const bool ARG1##_requires_grad                                             \
  ) {                                                                           \
    context->save_for_backward({gradient_output, ARG1});                        \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] = ARG1##_requires_grad;         \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    at::Tensor gradient_##ARG1 = c10::Dispatcher::singleton()                   \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward",                            \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        gradient_output,                                                        \
        ARG1                                                                    \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_##ARG1                                                           \
    };                                                                          \
  }                                                                             \
                                                                                \
  static std::vector<at::Tensor> backward(                                      \
    torch::autograd::AutogradContext* context,                                  \
    const std::vector<at::Tensor> &gradient_outputs                             \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    const bool ARG1##_requires_grad =                                           \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    const bool gradient_##ARG1##_defined = gradient_outputs[0].defined();       \
                                                                                \
    if (!(gradient_##ARG1##_defined && ARG1##_requires_grad)) {                 \
      return {                                                                  \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor()                                                            \
      };                                                                        \
    }                                                                           \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    at::Tensor gradient_gradient_##ARG1##_input;                                \
                                                                                \
    if (gradient_##ARG1##_defined && ARG1##_requires_grad) {                    \
      gradient_gradient_##ARG1##_input = gradient_outputs[0];                   \
    }                                                                           \
                                                                                \
    auto [                                                                      \
      gradient_gradient_output,                                                 \
      gradient_##ARG1                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward_backward",                   \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_gradient_##ARG1##_input,                                       \
        saved[0],                                                               \
        saved[1]                                                                \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      at::Tensor()                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
class CLASS_NAME                                                                \
  : public torch::autograd::Function<CLASS_NAME> {                              \
public:                                                                         \
  static at::Tensor forward(                                                    \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& ARG1                                                      \
  ) {                                                                           \
    context->save_for_backward({ARG1});                                         \
                                                                                \
    const bool condition = isFloatingType(ARG1.scalar_type()) ||                \
                           isComplexType(ARG1.scalar_type());                   \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] =                               \
      ARG1.requires_grad() && condition;                                        \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME,                                        \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        ARG1                                                                    \
      );                                                                        \
  }                                                                             \
                                                                                \
  static torch::autograd::variable_list backward(                               \
    torch::autograd::AutogradContext* context,                                  \
    const torch::autograd::variable_list &gradient_outputs                      \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    at::Tensor ARG1 = saved[0];                                                 \
                                                                                \
    at::Tensor gradient_output = gradient_outputs[0];                           \
                                                                                \
    bool ARG1##_requires_grad =                                                 \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    std::vector<at::Tensor> gradients = CLASS_NAME##Backward::apply(            \
      gradient_output,                                                          \
      ARG1,                                                                     \
      ARG1##_requires_grad                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG1;                                                 \
                                                                                \
    if (ARG1##_requires_grad) {                                                 \
      gradient_##ARG1 = gradients[0];                                           \
    } else {                                                                    \
      gradient_##ARG1 = at::Tensor();                                           \
    }                                                                           \
                                                                                \
    return {                                                                    \
      gradient_##ARG1                                                           \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1                                                        \
) {                                                                             \
  return CLASS_NAME::apply(                                                     \
    ARG1                                                                        \
  );                                                                            \
}                                                                               \
                                                                                \
}                                                                               \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                            \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::autograd::NAMESPACE::OPERATOR_NAME                           \
  );                                                                            \
}

#define AUTOGRAD_BINARY_OPERATOR(                                               \
  NAMESPACE,                                                                    \
  CLASS_NAME,                                                                   \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autograd::NAMESPACE {                                   \
                                                                                \
class CLASS_NAME##Backward                                                      \
  : public torch::autograd::Function<CLASS_NAME##Backward> {                    \
public:                                                                         \
  static std::vector<at::Tensor> forward(                                       \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const bool ARG1##_requires_grad                                             \
  ) {                                                                           \
    context->save_for_backward({gradient_output, ARG1, ARG2});                  \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] = ARG1##_requires_grad;         \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    auto [                                                                      \
      gradient_##ARG1,                                                          \
      gradient_##ARG2                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward",                            \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        gradient_output,                                                        \
        ARG1,                                                                   \
        ARG2                                                                    \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradient_##ARG2                                                           \
    };                                                                          \
  }                                                                             \
                                                                                \
  static std::vector<at::Tensor> backward(                                      \
    torch::autograd::AutogradContext* context,                                  \
    const std::vector<at::Tensor> &gradient_outputs                             \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    const bool ARG1##_requires_grad =                                           \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    const bool gradient_##ARG1##_defined = gradient_outputs[0].defined();       \
    const bool gradient_##ARG2##_defined = gradient_outputs[1].defined();       \
                                                                                \
    if (!(gradient_##ARG1##_defined && ARG1##_requires_grad) &&                 \
        !gradient_##ARG2##_defined) {                                           \
      return {                                                                  \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor()                                                            \
      };                                                                        \
    }                                                                           \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    at::Tensor gradient_gradient_##ARG1##_input;                                \
                                                                                \
    if (gradient_##ARG1##_defined && ARG1##_requires_grad) {                    \
      gradient_gradient_##ARG1##_input = gradient_outputs[0];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG2##_input;                                \
                                                                                \
    if (gradient_##ARG2##_defined) {                                            \
      gradient_gradient_##ARG2##_input = gradient_outputs[1];                   \
    }                                                                           \
                                                                                \
    auto [                                                                      \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward_backward",                   \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_gradient_##ARG1##_input,                                       \
        gradient_gradient_##ARG2##_input,                                       \
        saved[0],                                                               \
        saved[1],                                                               \
        saved[2]                                                                \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      at::Tensor()                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
class CLASS_NAME                                                                \
  : public torch::autograd::Function<CLASS_NAME> {                              \
public:                                                                         \
  static at::Tensor forward(                                                    \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2                                                      \
  ) {                                                                           \
    context->save_for_backward({ARG1, ARG2});                                   \
                                                                                \
    const bool condition = isFloatingType(ARG1.scalar_type()) ||                \
                           isComplexType(ARG1.scalar_type());                   \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] =                               \
      ARG1.requires_grad() && condition;                                        \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME,                                        \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        ARG1,                                                                   \
        ARG2                                                                    \
      );                                                                        \
  }                                                                             \
                                                                                \
  static torch::autograd::variable_list backward(                               \
    torch::autograd::AutogradContext* context,                                  \
    const torch::autograd::variable_list &gradient_outputs                      \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    at::Tensor ARG1 = saved[0];                                                 \
    at::Tensor ARG2 = saved[1];                                                 \
                                                                                \
    at::Tensor gradient_output = gradient_outputs[0];                           \
                                                                                \
    bool ARG1##_requires_grad =                                                 \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    std::vector<at::Tensor> gradients = CLASS_NAME##Backward::apply(            \
      gradient_output,                                                          \
      ARG1,                                                                     \
      ARG2,                                                                     \
      ARG1##_requires_grad                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG1;                                                 \
                                                                                \
    if (ARG1##_requires_grad) {                                                 \
      gradient_##ARG1 = gradients[0];                                           \
    } else {                                                                    \
      gradient_##ARG1 = at::Tensor();                                           \
    }                                                                           \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradients[1]                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2                                                        \
) {                                                                             \
  return CLASS_NAME::apply(                                                     \
    ARG1,                                                                       \
    ARG2                                                                        \
  );                                                                            \
}                                                                               \
                                                                                \
}                                                                               \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                            \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::autograd::NAMESPACE::OPERATOR_NAME                           \
  );                                                                            \
}

#define AUTOGRAD_TERNARY_OPERATOR(                                              \
  NAMESPACE,                                                                    \
  CLASS_NAME,                                                                   \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autograd::NAMESPACE {                                   \
                                                                                \
class CLASS_NAME##Backward                                                      \
  : public torch::autograd::Function<CLASS_NAME##Backward> {                    \
public:                                                                         \
  static std::vector<at::Tensor> forward(                                       \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const bool ARG1##_requires_grad                                             \
  ) {                                                                           \
    context->save_for_backward({gradient_output, ARG1, ARG2, ARG3});            \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] = ARG1##_requires_grad;         \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    auto [                                                                      \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward",                            \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        gradient_output,                                                        \
        ARG1,                                                                   \
        ARG2,                                                                   \
        ARG3                                                                    \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3                                                           \
    };                                                                          \
  }                                                                             \
                                                                                \
  static std::vector<at::Tensor> backward(                                      \
    torch::autograd::AutogradContext* context,                                  \
    const std::vector<at::Tensor> &gradient_outputs                             \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    const bool ARG1##_requires_grad =                                           \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    const bool gradient_##ARG1##_defined = gradient_outputs[0].defined();       \
    const bool gradient_##ARG2##_defined = gradient_outputs[1].defined();       \
    const bool gradient_##ARG3##_defined = gradient_outputs[2].defined();       \
                                                                                \
    if (!(gradient_##ARG1##_defined && ARG1##_requires_grad) &&                 \
        !gradient_##ARG2##_defined &&                                           \
        !gradient_##ARG3##_defined) {                                           \
      return {                                                                  \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor()                                                            \
      };                                                                        \
    }                                                                           \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    at::Tensor gradient_gradient_##ARG1##_input;                                \
                                                                                \
    if (gradient_##ARG1##_defined && ARG1##_requires_grad) {                    \
      gradient_gradient_##ARG1##_input = gradient_outputs[0];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG2##_input;                                \
                                                                                \
    if (gradient_##ARG2##_defined) {                                            \
      gradient_gradient_##ARG2##_input = gradient_outputs[1];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG3##_input;                                \
                                                                                \
    if (gradient_##ARG3##_defined) {                                            \
      gradient_gradient_##ARG3##_input = gradient_outputs[2];                   \
    }                                                                           \
                                                                                \
    auto [                                                                      \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward_backward",                   \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_gradient_##ARG1##_input,                                       \
        gradient_gradient_##ARG2##_input,                                       \
        gradient_gradient_##ARG3##_input,                                       \
        saved[0],                                                               \
        saved[1],                                                               \
        saved[2],                                                               \
        saved[3]                                                                \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3,                                                          \
      at::Tensor()                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
class CLASS_NAME                                                                \
  : public torch::autograd::Function<CLASS_NAME> {                              \
public:                                                                         \
  static at::Tensor forward(                                                    \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3                                                      \
  ) {                                                                           \
    context->save_for_backward({ARG1, ARG2, ARG3});                             \
                                                                                \
    const bool condition = isFloatingType(ARG1.scalar_type()) ||                \
                           isComplexType(ARG1.scalar_type());                   \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] =                               \
      ARG1.requires_grad() && condition;                                        \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME,                                        \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        ARG1,                                                                   \
        ARG2,                                                                   \
        ARG3                                                                    \
      );                                                                        \
  }                                                                             \
                                                                                \
  static torch::autograd::variable_list backward(                               \
    torch::autograd::AutogradContext* context,                                  \
    const torch::autograd::variable_list &gradient_outputs                      \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    at::Tensor ARG1 = saved[0];                                                 \
    at::Tensor ARG2 = saved[1];                                                 \
    at::Tensor ARG3 = saved[2];                                                 \
                                                                                \
    at::Tensor gradient_output = gradient_outputs[0];                           \
                                                                                \
    bool ARG1##_requires_grad =                                                 \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    std::vector<at::Tensor> gradients = CLASS_NAME##Backward::apply(            \
      gradient_output,                                                          \
      ARG1,                                                                     \
      ARG2,                                                                     \
      ARG3,                                                                     \
      ARG1##_requires_grad                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG1;                                                 \
                                                                                \
    if (ARG1##_requires_grad) {                                                 \
      gradient_##ARG1 = gradients[0];                                           \
    } else {                                                                    \
      gradient_##ARG1 = at::Tensor();                                           \
    }                                                                           \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradients[1],                                                             \
      gradients[2]                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3                                                        \
) {                                                                             \
  return CLASS_NAME::apply(                                                     \
    ARG1,                                                                       \
    ARG2,                                                                       \
    ARG3                                                                        \
  );                                                                            \
}                                                                               \
                                                                                \
}                                                                               \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                            \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::autograd::NAMESPACE::OPERATOR_NAME                           \
  );                                                                            \
}

#define AUTOGRAD_QUATERNARY_OPERATOR(                                           \
  NAMESPACE,                                                                    \
  CLASS_NAME,                                                                   \
  OPERATOR_NAME,                                                                \
  ARG1,                                                                         \
  ARG2,                                                                         \
  ARG3,                                                                         \
  ARG4                                                                          \
)                                                                               \
                                                                                \
namespace torchscience::autograd::NAMESPACE {                                   \
                                                                                \
class CLASS_NAME##Backward                                                      \
  : public torch::autograd::Function<CLASS_NAME##Backward> {                    \
public:                                                                         \
  static std::vector<at::Tensor> forward(                                       \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& gradient_output,                                          \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const at::Tensor& ARG4,                                                     \
    const bool ARG1##_requires_grad                                             \
  ) {                                                                           \
    context->save_for_backward({gradient_output, ARG1, ARG2, ARG3, ARG4});      \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] = ARG1##_requires_grad;         \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    auto [                                                                      \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3,                                                          \
      gradient_##ARG4                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward",                            \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>()                                                                      \
      .call(                                                                    \
        gradient_output,                                                        \
        ARG1,                                                                   \
        ARG2,                                                                   \
        ARG3,                                                                   \
        ARG4                                                                    \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3,                                                          \
      gradient_##ARG4                                                           \
    };                                                                          \
  }                                                                             \
                                                                                \
  static std::vector<at::Tensor> backward(                                      \
    torch::autograd::AutogradContext* context,                                  \
    const std::vector<at::Tensor> &gradient_outputs                             \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    const bool ARG1##_requires_grad =                                           \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    const bool gradient_##ARG1##_defined = gradient_outputs[0].defined();       \
    const bool gradient_##ARG2##_defined = gradient_outputs[1].defined();       \
    const bool gradient_##ARG3##_defined = gradient_outputs[2].defined();       \
    const bool gradient_##ARG4##_defined = gradient_outputs[3].defined();       \
                                                                                \
    if (!(gradient_##ARG1##_defined && ARG1##_requires_grad) &&                 \
        !gradient_##ARG2##_defined &&                                           \
        !gradient_##ARG3##_defined &&                                           \
        !gradient_##ARG4##_defined) {                                           \
      return {                                                                  \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor(),                                                           \
        at::Tensor()                                                            \
      };                                                                        \
    }                                                                           \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    at::Tensor gradient_gradient_##ARG1##_input;                                \
                                                                                \
    if (gradient_##ARG1##_defined && ARG1##_requires_grad) {                    \
      gradient_gradient_##ARG1##_input = gradient_outputs[0];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG2##_input;                                \
                                                                                \
    if (gradient_##ARG2##_defined) {                                            \
      gradient_gradient_##ARG2##_input = gradient_outputs[1];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG3##_input;                                \
                                                                                \
    if (gradient_##ARG3##_defined) {                                            \
      gradient_gradient_##ARG3##_input = gradient_outputs[2];                   \
    }                                                                           \
                                                                                \
    at::Tensor gradient_gradient_##ARG4##_input;                                \
                                                                                \
    if (gradient_##ARG4##_defined) {                                            \
      gradient_gradient_##ARG4##_input = gradient_outputs[3];                   \
    }                                                                           \
                                                                                \
    auto [                                                                      \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3,                                                          \
      gradient_##ARG4                                                           \
    ] = c10::Dispatcher::singleton()                                            \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME "_backward_backward",                   \
        ""                                                                      \
      )                                                                         \
      .typed<std::tuple<                                                        \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor,                                                             \
        at::Tensor                                                              \
      >(                                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        gradient_gradient_##ARG1##_input,                                       \
        gradient_gradient_##ARG2##_input,                                       \
        gradient_gradient_##ARG3##_input,                                       \
        gradient_gradient_##ARG4##_input,                                       \
        saved[0],                                                               \
        saved[1],                                                               \
        saved[2],                                                               \
        saved[3],                                                               \
        saved[4]                                                                \
      );                                                                        \
                                                                                \
    return {                                                                    \
      gradient_gradient_output,                                                 \
      gradient_##ARG1,                                                          \
      gradient_##ARG2,                                                          \
      gradient_##ARG3,                                                          \
      gradient_##ARG4,                                                          \
      at::Tensor()                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
class CLASS_NAME                                                                \
  : public torch::autograd::Function<CLASS_NAME> {                              \
public:                                                                         \
  static at::Tensor forward(                                                    \
    torch::autograd::AutogradContext* context,                                  \
    const at::Tensor& ARG1,                                                     \
    const at::Tensor& ARG2,                                                     \
    const at::Tensor& ARG3,                                                     \
    const at::Tensor& ARG4                                                      \
  ) {                                                                           \
    context->save_for_backward({ARG1, ARG2, ARG3, ARG4});                       \
                                                                                \
    const bool condition = isFloatingType(ARG1.scalar_type()) ||                \
                           isComplexType(ARG1.scalar_type());                   \
                                                                                \
    context->saved_data[#ARG1 "_requires_grad"] =                               \
      ARG1.requires_grad() && condition;                                        \
                                                                                \
    at::AutoDispatchBelowAutograd guard;                                        \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
      .findSchemaOrThrow(                                                       \
        "torchscience::" #OPERATOR_NAME,                                        \
        ""                                                                      \
      )                                                                         \
      .typed<at::Tensor(                                                        \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&,                                                      \
        const at::Tensor&                                                       \
      )>().call(                                                                \
        ARG1,                                                                   \
        ARG2,                                                                   \
        ARG3,                                                                   \
        ARG4                                                                    \
      );                                                                        \
  }                                                                             \
                                                                                \
  static torch::autograd::variable_list backward(                               \
    torch::autograd::AutogradContext* context,                                  \
    const torch::autograd::variable_list &gradient_outputs                      \
  ) {                                                                           \
    const torch::autograd::variable_list saved = context->get_saved_variables();\
                                                                                \
    at::Tensor ARG1 = saved[0];                                                 \
    at::Tensor ARG2 = saved[1];                                                 \
    at::Tensor ARG3 = saved[2];                                                 \
    at::Tensor ARG4 = saved[3];                                                 \
                                                                                \
    at::Tensor gradient_output = gradient_outputs[0];                           \
                                                                                \
    bool ARG1##_requires_grad =                                                 \
      context->saved_data[#ARG1 "_requires_grad"].toBool();                     \
                                                                                \
    std::vector<at::Tensor> gradients = CLASS_NAME##Backward::apply(            \
      gradient_output,                                                          \
      ARG1,                                                                     \
      ARG2,                                                                     \
      ARG3,                                                                     \
      ARG4,                                                                     \
      ARG1##_requires_grad                                                      \
    );                                                                          \
                                                                                \
    at::Tensor gradient_##ARG1;                                                 \
                                                                                \
    if (ARG1##_requires_grad) {                                                 \
      gradient_##ARG1 = gradients[0];                                           \
    } else {                                                                    \
      gradient_##ARG1 = at::Tensor();                                           \
    }                                                                           \
                                                                                \
    return {                                                                    \
      gradient_##ARG1,                                                          \
      gradients[1],                                                             \
      gradients[2],                                                             \
      gradients[3]                                                              \
    };                                                                          \
  }                                                                             \
};                                                                              \
                                                                                \
inline at::Tensor OPERATOR_NAME(                                                \
  const at::Tensor& ARG1,                                                       \
  const at::Tensor& ARG2,                                                       \
  const at::Tensor& ARG3,                                                       \
  const at::Tensor& ARG4                                                        \
) {                                                                             \
  return CLASS_NAME::apply(                                                     \
    ARG1,                                                                       \
    ARG2,                                                                       \
    ARG3,                                                                       \
    ARG4                                                                        \
  );                                                                            \
}                                                                               \
                                                                                \
}                                                                               \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {                            \
  module.impl(                                                                  \
    #OPERATOR_NAME,                                                             \
    &torchscience::autograd::NAMESPACE::OPERATOR_NAME                           \
  );                                                                            \
}
