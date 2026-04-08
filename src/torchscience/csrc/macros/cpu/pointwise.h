#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/library.h>

// =============================================================================
// CPU Pointwise Macros — Modular
//
// Signature: TORCHSCIENCE_CPU_POINTWISE_{ARITY}(category, complex, name, ...)
//   category: kernel namespace suffix (e.g. special_functions)
//   complex:  true → AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2
//             false → AT_DISPATCH_FLOATING_TYPES_AND2
//   name:     operator name
//   argN:     argument names
// =============================================================================

// ---------------------------------------------------------------------------
// UNARY — complex = true
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_true(category, name, arg1) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline at::Tensor name##_backward(                                         \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  at::Tensor gradient_output;                                              \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_output)                                           \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(        \
  const at::Tensor &gradient_gradient_input,                               \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  if (!gradient_gradient_input.defined()) {                                \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor gradient_output;                                              \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(gradient_output)                                           \
    .add_const_input(gradient_gradient_input)                               \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient_gradient,                                      \
          scalar_t gradient,                                               \
          scalar_t arg1                                                    \
        ) -> std::tuple<                                                   \
          scalar_t,                                                        \
          scalar_t                                                         \
        > {                                                                \
          return kernel::category::name##_backward_backward(               \
            gradient_gradient,                                             \
            gradient,                                                      \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// UNARY — complex = false
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_false(category, name, arg1) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline at::Tensor name##_backward(                                         \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  at::Tensor gradient_output;                                              \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_output)                                           \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(        \
  const at::Tensor &gradient_gradient_input,                               \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input                                           \
) {                                                                        \
  if (!gradient_gradient_input.defined()) {                                \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor gradient_output;                                              \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(gradient_output)                                           \
    .add_const_input(gradient_gradient_input)                               \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient_gradient,                                      \
          scalar_t gradient,                                               \
          scalar_t arg1                                                    \
        ) -> std::tuple<                                                   \
          scalar_t,                                                        \
          scalar_t                                                         \
        > {                                                                \
          return kernel::category::name##_backward_backward(               \
            gradient_gradient,                                             \
            gradient,                                                      \
            arg1                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// UNARY — entry point
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_UNARY(category, complex, name, arg1) \
    TORCHSCIENCE_CPU_POINTWISE_UNARY_DISPATCH_##complex(category, name, arg1)

// ===========================================================================
// BINARY — complex = true
// ===========================================================================

#define TORCHSCIENCE_CPU_POINTWISE_BINARY_DISPATCH_true(category, name, arg1, arg2) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor> name##_backward(                 \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> std::tuple<scalar_t, scalar_t> {                              \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward_backward(\
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {                    \
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// BINARY — complex = false
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_BINARY_DISPATCH_false(category, name, arg1, arg2) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor> name##_backward(                 \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> std::tuple<scalar_t, scalar_t> {                              \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward_backward(\
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {                    \
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// BINARY — entry point
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_BINARY(category, complex, name, arg1, arg2) \
    TORCHSCIENCE_CPU_POINTWISE_BINARY_DISPATCH_##complex(category, name, arg1, arg2)

// ===========================================================================
// TERNARY — complex = true
// ===========================================================================

#define TORCHSCIENCE_CPU_POINTWISE_TERNARY_DISPATCH_true(category, name, arg1, arg2, arg3) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward(     \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {                    \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<                                                         \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor                           \
> name##_backward_backward(                                                \
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &arg3##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined() &&                         \
      !arg3##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                \
    ? arg3##_gradient_gradient_input                                       \
    : at::zeros_like(arg3##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(arg3##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t arg3##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {          \
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            arg3##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// TERNARY — complex = false
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_TERNARY_DISPATCH_false(category, name, arg1, arg2, arg3) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> name##_backward(     \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t> {                    \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<                                                         \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor                           \
> name##_backward_backward(                                                \
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &arg3##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined() &&                         \
      !arg3##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                \
    ? arg3##_gradient_gradient_input                                       \
    : at::zeros_like(arg3##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(arg3##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t arg3##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {          \
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            arg3##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// TERNARY — entry point
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_TERNARY(category, complex, name, arg1, arg2, arg3) \
    TORCHSCIENCE_CPU_POINTWISE_TERNARY_DISPATCH_##complex(category, name, arg1, arg2, arg3)

// ===========================================================================
// QUATERNARY — complex = true
// ===========================================================================

#define TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_DISPATCH_true(category, name, arg1, arg2, arg3, arg4) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>          \
name##_backward(                                                           \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {          \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<                                                         \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor               \
> name##_backward_backward(                                                \
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &arg3##_gradient_gradient_input,                        \
  const at::Tensor &arg4##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined() &&                         \
      !arg3##_gradient_gradient_input.defined() &&                         \
      !arg4##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                \
    ? arg3##_gradient_gradient_input                                       \
    : at::zeros_like(arg3##_input);                                        \
  auto arg4##_gg = arg4##_gradient_gradient_input.defined()                \
    ? arg4##_gradient_gradient_input                                       \
    : at::zeros_like(arg4##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(arg3##_gg)                                            \
    .add_const_input(arg4##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(                             \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t arg3##_gradient_gradient,                               \
          scalar_t arg4##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> {\
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            arg3##_gradient_gradient,                                      \
            arg4##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3),                                                    \
    iterator.output(4)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// QUATERNARY — complex = false
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_DISPATCH_false(category, name, arg1, arg2, arg3, arg4) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>          \
name##_backward(                                                           \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {          \
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<                                                         \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor               \
> name##_backward_backward(                                                \
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &arg3##_gradient_gradient_input,                        \
  const at::Tensor &arg4##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined() &&                         \
      !arg3##_gradient_gradient_input.defined() &&                         \
      !arg4##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                \
    ? arg3##_gradient_gradient_input                                       \
    : at::zeros_like(arg3##_input);                                        \
  auto arg4##_gg = arg4##_gradient_gradient_input.defined()                \
    ? arg4##_gradient_gradient_input                                       \
    : at::zeros_like(arg4##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(arg3##_gg)                                            \
    .add_const_input(arg4##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t arg3##_gradient_gradient,                               \
          scalar_t arg4##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> {\
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            arg3##_gradient_gradient,                                      \
            arg4##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3),                                                    \
    iterator.output(4)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}

// ---------------------------------------------------------------------------
// QUATERNARY — entry point
// ---------------------------------------------------------------------------

#define TORCHSCIENCE_CPU_POINTWISE_QUATERNARY(category, complex, name, arg1, arg2, arg3, arg4) \
    TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_DISPATCH_##complex(category, name, arg1, arg2, arg3, arg4)

// ===========================================================================
// QUINARY — complex = false only (no complex quinary in the existing codebase)
// ===========================================================================

#define TORCHSCIENCE_CPU_POINTWISE_QUINARY(category, complex, name, arg1, arg2, arg3, arg4, arg5) \
namespace torchscience::cpu::category {                                    \
                                                                           \
inline at::Tensor name(                                                    \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input,                                          \
  const at::Tensor &arg5##_input                                           \
) {                                                                        \
  at::Tensor output;                                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(output)                                                    \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .add_const_input(arg5##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name,                                                                 \
    [&] {                                                                  \
      at::native::cpu_kernel(                                              \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4,                                                   \
          scalar_t arg5                                                    \
        ) -> scalar_t {                                                    \
          return kernel::category::name(                                   \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4,                                                          \
            arg5                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return iterator.output();                                                \
}                                                                          \
                                                                           \
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> \
name##_backward(                                                           \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input,                                          \
  const at::Tensor &arg5##_input                                           \
) {                                                                        \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
  at::Tensor arg5##_gradient_output;                                       \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_output(arg5##_gradient_output)                                    \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .add_const_input(arg5##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward",                                                     \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4,                                                   \
          scalar_t arg5                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> {\
          return kernel::category::name##_backward(                        \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4,                                                          \
            arg5                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3),                                                    \
    iterator.output(4)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
inline std::tuple<                                                         \
  at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor   \
> name##_backward_backward(                                                \
  const at::Tensor &arg1##_gradient_gradient_input,                        \
  const at::Tensor &arg2##_gradient_gradient_input,                        \
  const at::Tensor &arg3##_gradient_gradient_input,                        \
  const at::Tensor &arg4##_gradient_gradient_input,                        \
  const at::Tensor &arg5##_gradient_gradient_input,                        \
  const at::Tensor &gradient_input,                                        \
  const at::Tensor &arg1##_input,                                          \
  const at::Tensor &arg2##_input,                                          \
  const at::Tensor &arg3##_input,                                          \
  const at::Tensor &arg4##_input,                                          \
  const at::Tensor &arg5##_input                                           \
) {                                                                        \
  if (!arg1##_gradient_gradient_input.defined() &&                         \
      !arg2##_gradient_gradient_input.defined() &&                         \
      !arg3##_gradient_gradient_input.defined() &&                         \
      !arg4##_gradient_gradient_input.defined() &&                         \
      !arg5##_gradient_gradient_input.defined()) {                         \
    return {                                                               \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor(),                                                        \
      at::Tensor()                                                         \
    };                                                                     \
  }                                                                        \
                                                                           \
  at::Tensor gradient_gradient_output;                                     \
  at::Tensor arg1##_gradient_output;                                       \
  at::Tensor arg2##_gradient_output;                                       \
  at::Tensor arg3##_gradient_output;                                       \
  at::Tensor arg4##_gradient_output;                                       \
  at::Tensor arg5##_gradient_output;                                       \
                                                                           \
  auto arg1##_gg = arg1##_gradient_gradient_input.defined()                \
    ? arg1##_gradient_gradient_input                                       \
    : at::zeros_like(arg1##_input);                                        \
  auto arg2##_gg = arg2##_gradient_gradient_input.defined()                \
    ? arg2##_gradient_gradient_input                                       \
    : at::zeros_like(arg2##_input);                                        \
  auto arg3##_gg = arg3##_gradient_gradient_input.defined()                \
    ? arg3##_gradient_gradient_input                                       \
    : at::zeros_like(arg3##_input);                                        \
  auto arg4##_gg = arg4##_gradient_gradient_input.defined()                \
    ? arg4##_gradient_gradient_input                                       \
    : at::zeros_like(arg4##_input);                                        \
  auto arg5##_gg = arg5##_gradient_gradient_input.defined()                \
    ? arg5##_gradient_gradient_input                                       \
    : at::zeros_like(arg5##_input);                                        \
                                                                           \
  auto iterator = at::TensorIteratorConfig()                               \
    .add_output(gradient_gradient_output)                                   \
    .add_output(arg1##_gradient_output)                                    \
    .add_output(arg2##_gradient_output)                                    \
    .add_output(arg3##_gradient_output)                                    \
    .add_output(arg4##_gradient_output)                                    \
    .add_output(arg5##_gradient_output)                                    \
    .add_const_input(arg1##_gg)                                            \
    .add_const_input(arg2##_gg)                                            \
    .add_const_input(arg3##_gg)                                            \
    .add_const_input(arg4##_gg)                                            \
    .add_const_input(arg5##_gg)                                            \
    .add_const_input(gradient_input)                                       \
    .add_const_input(arg1##_input)                                         \
    .add_const_input(arg2##_input)                                         \
    .add_const_input(arg3##_input)                                         \
    .add_const_input(arg4##_input)                                         \
    .add_const_input(arg5##_input)                                         \
    .promote_inputs_to_common_dtype(true)                                  \
    .cast_common_dtype_to_outputs(true)                                    \
    .build();                                                              \
                                                                           \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                         \
    at::kBFloat16,                                                         \
    at::kHalf,                                                             \
    iterator.common_dtype(),                                               \
    #name "_backward_backward",                                            \
    [&] {                                                                  \
      at::native::cpu_kernel_multiple_outputs(                             \
        iterator,                                                          \
        [] (                                                               \
          scalar_t arg1##_gradient_gradient,                               \
          scalar_t arg2##_gradient_gradient,                               \
          scalar_t arg3##_gradient_gradient,                               \
          scalar_t arg4##_gradient_gradient,                               \
          scalar_t arg5##_gradient_gradient,                               \
          scalar_t gradient,                                               \
          scalar_t arg1,                                                   \
          scalar_t arg2,                                                   \
          scalar_t arg3,                                                   \
          scalar_t arg4,                                                   \
          scalar_t arg5                                                    \
        ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> { \
          return kernel::category::name##_backward_backward(               \
            arg1##_gradient_gradient,                                      \
            arg2##_gradient_gradient,                                      \
            arg3##_gradient_gradient,                                      \
            arg4##_gradient_gradient,                                      \
            arg5##_gradient_gradient,                                      \
            gradient,                                                      \
            arg1,                                                          \
            arg2,                                                          \
            arg3,                                                          \
            arg4,                                                          \
            arg5                                                           \
          );                                                               \
        }                                                                  \
      );                                                                   \
    }                                                                      \
  );                                                                       \
                                                                           \
  return {                                                                 \
    iterator.output(0),                                                    \
    iterator.output(1),                                                    \
    iterator.output(2),                                                    \
    iterator.output(3),                                                    \
    iterator.output(4),                                                    \
    iterator.output(5)                                                     \
  };                                                                       \
}                                                                          \
                                                                           \
} /* namespace torchscience::cpu::category */                              \
                                                                           \
TORCH_LIBRARY_IMPL(torchscience, CPU, module) {                            \
  module.impl(                                                             \
    #name,                                                                 \
    torchscience::cpu::category::name                                      \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward",                                                     \
    torchscience::cpu::category::name##_backward                           \
  );                                                                       \
                                                                           \
  module.impl(                                                             \
    #name "_backward_backward",                                            \
    torchscience::cpu::category::name##_backward_backward                  \
  );                                                                       \
}
