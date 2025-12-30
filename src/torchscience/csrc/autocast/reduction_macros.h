// src/torchscience/csrc/autocast/reduction_macros.h
#pragma once

#include <tuple>

#include <ATen/autocast_mode.h>
#include <torch/library.h>

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autocast)
// =============================================================================

#define TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)  \
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, at::OptionalIntArrayRef, bool                    \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(                                                      \
            const at::Tensor&, const at::Tensor&,                               \
            at::OptionalIntArrayRef, bool                                       \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&,            \
            at::OptionalIntArrayRef, bool                                       \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_grad_input),             \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg),                         \
            dim,                                                                \
            keepdim                                                             \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}

// Helper macros
#define TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(...) __VA_ARGS__
#define TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(...) __VA_ARGS__

// =============================================================================
// FIXED REDUCTION MACROS (Autocast)
// =============================================================================

#define TORCHSCIENCE_AUTOCAST_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, arg, ...)  \
namespace torchscience::autocast::NS {                                          \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name, "")                          \
        .typed<at::Tensor(const at::Tensor&                                     \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
inline at::Tensor name##_backward(                                              \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward", "")              \
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&                  \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
inline std::tuple<at::Tensor, at::Tensor> name##_backward_backward(             \
    const at::Tensor& grad_grad_input,                                          \
    const at::Tensor& grad_output,                                              \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    c10::impl::ExcludeDispatchKeyGuard no_autocast(                             \
        c10::DispatchKey::Autocast                                              \
    );                                                                          \
                                                                                \
    return c10::Dispatcher::singleton()                                         \
        .findSchemaOrThrow("torchscience::" #name "_backward_backward", "")     \
        .typed<std::tuple<at::Tensor, at::Tensor>(                              \
            const at::Tensor&, const at::Tensor&, const at::Tensor&             \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
        )>()                                                                    \
        .call(                                                                  \
            at::autocast::cached_cast(at::kFloat, grad_grad_input),             \
            at::autocast::cached_cast(at::kFloat, grad_output),                 \
            at::autocast::cached_cast(at::kFloat, arg)                          \
            __VA_OPT__(, TORCHSCIENCE_AUTOCAST_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
}                                                                               \
                                                                                \
} /* namespace torchscience::autocast::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {                                 \
    m.impl(#name, &torchscience::autocast::NS::name);                           \
    m.impl(#name "_backward", &torchscience::autocast::NS::name##_backward);    \
    m.impl(#name "_backward_backward", &torchscience::autocast::NS::name##_backward_backward);\
}
