// src/torchscience/csrc/autograd/reduction_macros.h
#pragma once

#include <vector>

#include <torch/extension.h>

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autograd)
// =============================================================================

#define TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg, ...)\
namespace torchscience::autograd::NS {                                          \
                                                                                \
class Name##Backward : public torch::autograd::Function<Name##Backward> {       \
public:                                                                         \
    static std::vector<at::Tensor> forward(                                     \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& grad_output,                                          \
        const at::Tensor& arg,                                                  \
        std::vector<int64_t> dim_vec,                                           \
        bool keepdim,                                                           \
        bool arg##_requires_grad                                                \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name "_backward", "")          \
            .typed<at::Tensor(                                                  \
                const at::Tensor&, const at::Tensor&,                           \
                at::OptionalIntArrayRef, bool                                   \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg, dim_ref, keepdim                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)))};     \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};\
        }                                                                       \
                                                                                \
        auto dim_vec = ctx->saved_data["dim"].toIntVector();                    \
        bool keepdim = ctx->saved_data["keepdim"].toBool();                     \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        auto [gg_output, new_grad] = c10::Dispatcher::singleton()               \
            .findSchemaOrThrow("torchscience::" #name "_backward_backward", "") \
            .typed<std::tuple<at::Tensor, at::Tensor>(                          \
                const at::Tensor&, const at::Tensor&, const at::Tensor&,        \
                at::OptionalIntArrayRef, bool                                   \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1], dim_ref, keepdim         \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)));\
                                                                                \
        return {gg_output, new_grad, at::Tensor(), at::Tensor(), at::Tensor()}; \
    }                                                                           \
};                                                                              \
                                                                                \
class Name : public torch::autograd::Function<Name> {                           \
public:                                                                         \
    static at::Tensor forward(                                                  \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& arg,                                                  \
        std::vector<int64_t> dim_vec,                                           \
        bool keepdim                                                            \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        at::OptionalIntArrayRef dim_ref = dim_vec.empty()                       \
            ? at::OptionalIntArrayRef()                                         \
            : at::OptionalIntArrayRef(dim_vec);                                 \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name, "")                      \
            .typed<at::Tensor(                                                  \
                const at::Tensor&, at::OptionalIntArrayRef, bool                \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>();                                                               \
                                                                                \
        return op.call(arg, dim_ref, keepdim                                    \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)));\
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
        torch::autograd::AutogradContext* ctx,                                  \
        const torch::autograd::variable_list& grad_outputs                      \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        auto dim_vec = ctx->saved_data["dim"].toIntVector();                    \
        bool keepdim = ctx->saved_data["keepdim"].toBool();                     \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], dim_vec, keepdim, arg##_requires_grad    \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
                                                                                \
        return {                                                                \
            arg##_requires_grad ? grads[0] : at::Tensor(),                      \
            at::Tensor(),  /* dim */                                            \
            at::Tensor()   /* keepdim */                                        \
            /* extra args return at::Tensor() */                                \
        };                                                                      \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    std::vector<int64_t> dim_vec;                                               \
    if (dim.has_value()) {                                                      \
        dim_vec = dim->vec();                                                   \
    }                                                                           \
    return Name::apply(arg, dim_vec, keepdim                                    \
        __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__))); \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}

// Helper macros for extracting types and names from variadic args
// Note: These are simplified - for complex extra args, users may need
// to define their own specialized macros
#define TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(...) __VA_ARGS__
#define TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(...) __VA_ARGS__

// =============================================================================
// FIXED REDUCTION MACROS (Autograd)
// =============================================================================

#define TORCHSCIENCE_AUTOGRAD_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg, ...)\
namespace torchscience::autograd::NS {                                          \
                                                                                \
class Name##Backward : public torch::autograd::Function<Name##Backward> {       \
public:                                                                         \
    static std::vector<at::Tensor> forward(                                     \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& grad_output,                                          \
        const at::Tensor& arg,                                                  \
        bool arg##_requires_grad                                                \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name "_backward", "")          \
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&              \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg                                        \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)))};     \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor()};                  \
        }                                                                       \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        auto [gg_output, new_grad] = c10::Dispatcher::singleton()               \
            .findSchemaOrThrow("torchscience::" #name "_backward_backward", "") \
            .typed<std::tuple<at::Tensor, at::Tensor>(                          \
                const at::Tensor&, const at::Tensor&, const at::Tensor&         \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1]                           \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)));\
                                                                                \
        return {gg_output, new_grad, at::Tensor()};                             \
    }                                                                           \
};                                                                              \
                                                                                \
class Name : public torch::autograd::Function<Name> {                           \
public:                                                                         \
    static at::Tensor forward(                                                  \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& arg                                                   \
        __VA_OPT__(, __VA_ARGS__)                                               \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name, "")                      \
            .typed<at::Tensor(const at::Tensor&                                 \
                __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_TYPES(__VA_ARGS__))\
            )>();                                                               \
                                                                                \
        return op.call(arg                                                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__)));      \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
        torch::autograd::AutogradContext* ctx,                                  \
        const torch::autograd::variable_list& grad_outputs                      \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], arg##_requires_grad                      \
            __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__))\
        );                                                                      \
                                                                                \
        return {arg##_requires_grad ? grads[0] : at::Tensor()};                 \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    __VA_OPT__(, __VA_ARGS__)                                                   \
) {                                                                             \
    return Name::apply(arg                                                      \
        __VA_OPT__(, TORCHSCIENCE_AUTOGRAD_REDUCTION_EXTRA_NAMES(__VA_ARGS__))); \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}
