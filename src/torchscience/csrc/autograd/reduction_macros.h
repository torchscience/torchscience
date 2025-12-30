// src/torchscience/csrc/autograd/reduction_macros.h
#pragma once

#include <vector>

#include <torch/extension.h>

// =============================================================================
// HELPER MACROS
// =============================================================================

// Use these to wrap extra parameters for _EX macros
// TSCI_EXTRA(bool fisher, bool bias) expands to: , bool fisher, bool bias
// TSCI_TYPES(bool, bool) expands to: , bool, bool
#ifndef TSCI_EXTRA
#define TSCI_EXTRA(...) , __VA_ARGS__
#endif
#ifndef TSCI_NO_EXTRA
#define TSCI_NO_EXTRA
#endif
#define TSCI_TYPES(...) , __VA_ARGS__
#define TSCI_NO_TYPES

// Macros for saving/loading extra params in autograd context
// Usage: TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ctx->saved_data["bias"] = bias;)
// Usage: TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); bool bias = ctx->saved_data["bias"].toBool();)
#define TSCI_SAVE(...) __VA_ARGS__
#define TSCI_LOAD(...) __VA_ARGS__
#define TSCI_NO_SAVE
#define TSCI_NO_LOAD

// Placeholder returns for extra params in backward (one at::Tensor() per extra param)
// Usage: TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor()) for 2 extra params
#define TSCI_GRAD_PLACEHOLDERS(...) , __VA_ARGS__
#define TSCI_NO_GRAD_PLACEHOLDERS

// =============================================================================
// DIM-BASED REDUCTION MACROS (Autograd)
// =============================================================================

/**
 * Autograd macro for unary dim-based reduction operators (no extra params).
 */
#define TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg) \
    TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, Name, arg, , , , , , )

/**
 * Autograd macro for unary dim-based reduction operators with extra parameters.
 *
 * @param NS Namespace suffix
 * @param name Operator name (lowercase, for dispatch)
 * @param Name Class name (PascalCase, for Function classes)
 * @param arg Tensor argument name
 * @param EXTRA_PARAMS Param declarations with leading comma: TSCI_EXTRA(bool fisher, bool bias)
 * @param EXTRA_ARGS Param names with leading comma: TSCI_EXTRA(fisher, bias)
 * @param EXTRA_TYPES Type list with leading comma: TSCI_TYPES(bool, bool)
 * @param EXTRA_SAVE Statements to save extra params: TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ...)
 * @param EXTRA_LOAD Statements to load extra params: TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); ...)
 * @param EXTRA_GRAD_PLACEHOLDERS Placeholder returns for backward: TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())
 *
 * Example:
 *   TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(
 *       statistics::descriptive, kurtosis, Kurtosis, input,
 *       TSCI_EXTRA(bool fisher, bool bias),
 *       TSCI_EXTRA(fisher, bias),
 *       TSCI_TYPES(bool, bool),
 *       TSCI_SAVE(ctx->saved_data["fisher"] = fisher; ctx->saved_data["bias"] = bias;),
 *       TSCI_LOAD(bool fisher = ctx->saved_data["fisher"].toBool(); bool bias = ctx->saved_data["bias"].toBool();),
 *       TSCI_GRAD_PLACEHOLDERS(at::Tensor(), at::Tensor())
 *   )
 */
#define TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR_EX(NS, name, Name, arg, EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES, EXTRA_SAVE, EXTRA_LOAD, EXTRA_GRAD_PLACEHOLDERS)\
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
        EXTRA_PARAMS                                                            \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
        EXTRA_SAVE                                                              \
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
                EXTRA_TYPES                                                     \
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg, dim_ref, keepdim                      \
            EXTRA_ARGS)};                                                       \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
                                                                                \
        auto dim_vec = ctx->saved_data["dim"].toIntVector();                    \
        bool keepdim = ctx->saved_data["keepdim"].toBool();                     \
        EXTRA_LOAD                                                              \
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()\
                EXTRA_GRAD_PLACEHOLDERS};                                       \
        }                                                                       \
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
                EXTRA_TYPES                                                     \
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1], dim_ref, keepdim         \
                EXTRA_ARGS);                                                    \
                                                                                \
        return {gg_output, new_grad, at::Tensor(), at::Tensor(), at::Tensor()   \
            EXTRA_GRAD_PLACEHOLDERS};                                           \
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
        EXTRA_PARAMS                                                            \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data["dim"] = dim_vec;                                       \
        ctx->saved_data["keepdim"] = keepdim;                                   \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
        EXTRA_SAVE                                                              \
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
                EXTRA_TYPES                                                     \
            )>();                                                               \
                                                                                \
        return op.call(arg, dim_ref, keepdim                                    \
            EXTRA_ARGS);                                                        \
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
        EXTRA_LOAD                                                              \
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], dim_vec, keepdim, arg##_requires_grad    \
            EXTRA_ARGS                                                          \
        );                                                                      \
                                                                                \
        return {                                                                \
            arg##_requires_grad ? grads[0] : at::Tensor(),                      \
            at::Tensor(),  /* dim */                                            \
            at::Tensor()   /* keepdim */                                        \
            EXTRA_GRAD_PLACEHOLDERS                                             \
        };                                                                      \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg,                                                      \
    at::OptionalIntArrayRef dim,                                                \
    bool keepdim                                                                \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    std::vector<int64_t> dim_vec;                                               \
    if (dim.has_value()) {                                                      \
        dim_vec = dim->vec();                                                   \
    }                                                                           \
    return Name::apply(arg, dim_vec, keepdim                                    \
        EXTRA_ARGS);                                                            \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}

// =============================================================================
// FIXED REDUCTION MACROS (Autograd)
// =============================================================================

/**
 * Autograd macro for unary fixed reduction operators (no extra params).
 */
#define TORCHSCIENCE_AUTOGRAD_FIXED_REDUCTION_UNARY_OPERATOR(NS, name, Name, arg) \
    TORCHSCIENCE_AUTOGRAD_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, Name, arg, , , , , , )

/**
 * Autograd macro for unary fixed reduction operators with extra parameters.
 *
 * @param NS Namespace suffix
 * @param name Operator name (lowercase, for dispatch)
 * @param Name Class name (PascalCase, for Function classes)
 * @param arg Tensor argument name
 * @param EXTRA_PARAMS Param declarations with leading comma
 * @param EXTRA_ARGS Param names with leading comma
 * @param EXTRA_TYPES Type list with leading comma
 * @param EXTRA_SAVE Statements to save extra params
 * @param EXTRA_LOAD Statements to load extra params
 * @param EXTRA_GRAD_PLACEHOLDERS Placeholder returns for backward
 */
#define TORCHSCIENCE_AUTOGRAD_FIXED_REDUCTION_UNARY_OPERATOR_EX(NS, name, Name, arg, EXTRA_PARAMS, EXTRA_ARGS, EXTRA_TYPES, EXTRA_SAVE, EXTRA_LOAD, EXTRA_GRAD_PLACEHOLDERS)\
namespace torchscience::autograd::NS {                                          \
                                                                                \
class Name##Backward : public torch::autograd::Function<Name##Backward> {       \
public:                                                                         \
    static std::vector<at::Tensor> forward(                                     \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& grad_output,                                          \
        const at::Tensor& arg,                                                  \
        bool arg##_requires_grad                                                \
        EXTRA_PARAMS                                                            \
    ) {                                                                         \
        ctx->save_for_backward({grad_output, arg});                             \
        ctx->saved_data[#arg "_requires_grad"] = arg##_requires_grad;           \
        EXTRA_SAVE                                                              \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name "_backward", "")          \
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&              \
                EXTRA_TYPES                                                     \
            )>();                                                               \
                                                                                \
        return {op.call(grad_output, arg                                        \
            EXTRA_ARGS)};                                                       \
    }                                                                           \
                                                                                \
    static std::vector<at::Tensor> backward(                                    \
        torch::autograd::AutogradContext* ctx,                                  \
        const std::vector<at::Tensor>& grad_outputs                             \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
        EXTRA_LOAD                                                              \
                                                                                \
        if (!grad_outputs[0].defined() || !arg##_requires_grad) {               \
            return {at::Tensor(), at::Tensor(), at::Tensor()                    \
                EXTRA_GRAD_PLACEHOLDERS};                                       \
        }                                                                       \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        auto [gg_output, new_grad] = c10::Dispatcher::singleton()               \
            .findSchemaOrThrow("torchscience::" #name "_backward_backward", "") \
            .typed<std::tuple<at::Tensor, at::Tensor>(                          \
                const at::Tensor&, const at::Tensor&, const at::Tensor&         \
                EXTRA_TYPES                                                     \
            )>()                                                                \
            .call(grad_outputs[0], saved[0], saved[1]                           \
                EXTRA_ARGS);                                                    \
                                                                                \
        return {gg_output, new_grad, at::Tensor()                               \
            EXTRA_GRAD_PLACEHOLDERS};                                           \
    }                                                                           \
};                                                                              \
                                                                                \
class Name : public torch::autograd::Function<Name> {                           \
public:                                                                         \
    static at::Tensor forward(                                                  \
        torch::autograd::AutogradContext* ctx,                                  \
        const at::Tensor& arg                                                   \
        EXTRA_PARAMS                                                            \
    ) {                                                                         \
        ctx->save_for_backward({arg});                                          \
        ctx->saved_data[#arg "_requires_grad"] = arg.requires_grad() &&         \
            (at::isFloatingType(arg.scalar_type()) ||                           \
             at::isComplexType(arg.scalar_type()));                             \
        EXTRA_SAVE                                                              \
                                                                                \
        at::AutoDispatchBelowAutograd guard;                                    \
                                                                                \
        static auto op = c10::Dispatcher::singleton()                           \
            .findSchemaOrThrow("torchscience::" #name, "")                      \
            .typed<at::Tensor(const at::Tensor&                                 \
                EXTRA_TYPES                                                     \
            )>();                                                               \
                                                                                \
        return op.call(arg                                                      \
            EXTRA_ARGS);                                                        \
    }                                                                           \
                                                                                \
    static torch::autograd::variable_list backward(                             \
        torch::autograd::AutogradContext* ctx,                                  \
        const torch::autograd::variable_list& grad_outputs                      \
    ) {                                                                         \
        auto saved = ctx->get_saved_variables();                                \
        bool arg##_requires_grad = ctx->saved_data[#arg "_requires_grad"].toBool();\
        EXTRA_LOAD                                                              \
                                                                                \
        auto grads = Name##Backward::apply(                                     \
            grad_outputs[0], saved[0], arg##_requires_grad                      \
            EXTRA_ARGS                                                          \
        );                                                                      \
                                                                                \
        return {arg##_requires_grad ? grads[0] : at::Tensor()                   \
            EXTRA_GRAD_PLACEHOLDERS};                                           \
    }                                                                           \
};                                                                              \
                                                                                \
inline at::Tensor name(                                                         \
    const at::Tensor& arg                                                       \
    EXTRA_PARAMS                                                                \
) {                                                                             \
    return Name::apply(arg                                                      \
        EXTRA_ARGS);                                                            \
}                                                                               \
                                                                                \
} /* namespace torchscience::autograd::NS */                                    \
                                                                                \
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {                                 \
    m.impl(#name, &torchscience::autograd::NS::name);                           \
}
