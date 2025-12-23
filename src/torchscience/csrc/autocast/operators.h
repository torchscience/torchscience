#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast {

// ============================================================================
// Helper function to determine target dtype for autocast
// ============================================================================

namespace detail {

inline at::ScalarType get_autocast_target_dtype(
    const at::Tensor& tensor,
    at::ScalarType base_dtype
) {
    if (isComplexType(tensor.scalar_type())) {
        if (base_dtype == at::kHalf) {
            return at::kComplexHalf;
        } else if (base_dtype == at::kBFloat16) {
            return at::kComplexFloat;
        } else {
            return at::kComplexFloat;
        }
    }
    return base_dtype;
}

template<typename... Tensors>
inline at::ScalarType get_autocast_dtype(const Tensors&... tensors) {
    // Check if any tensor is on CPU (all must be CPU for CPU autocast)
    bool all_cpu = (... && tensors.device().is_cpu());

    return all_cpu
        ? at::autocast::get_autocast_dtype(at::kCPU)
        : at::autocast::get_autocast_dtype(at::kCUDA);
}

template<typename... Tensors>
inline bool any_complex(const Tensors&... tensors) {
    return (... || isComplexType(tensors.scalar_type()));
}

template<typename... Tensors>
inline at::ScalarType get_target_dtype(const Tensors&... tensors) {
    at::ScalarType dtype = get_autocast_dtype(tensors...);

    if (any_complex(tensors...)) {
        if (dtype == at::kHalf) {
            return at::kComplexHalf;
        } else if (dtype == at::kBFloat16) {
            return at::kComplexFloat;
        } else {
            return at::kComplexFloat;
        }
    }
    return dtype;
}

}  // namespace detail

// ============================================================================
// AutocastUnaryOperator
// ============================================================================

struct AutocastUnaryOperator {
    template<typename Dispatcher>
    static at::Tensor forward(
        const at::Tensor& input,
        Dispatcher dispatcher
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype = detail::get_target_dtype(input);

        return dispatcher(at::autocast::cached_cast(target_dtype, input));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* schema_name
    ) {
        module.impl(name, [schema_name](const at::Tensor& input) {
            return forward(input, [schema_name](const at::Tensor& x) {
                return c10::Dispatcher::singleton()
                    .findSchemaOrThrow(schema_name, "")
                    .typed<at::Tensor(const at::Tensor&)>()
                    .call(x);
            });
        });
    }
};

#define REGISTER_AUTOCAST_UNARY(module, name) \
    ::torchscience::autocast::AutocastUnaryOperator::register_all( \
        module, #name, "torchscience::" #name)

// ============================================================================
// AutocastBinaryOperator
// ============================================================================

struct AutocastBinaryOperator {
    template<typename Dispatcher>
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        Dispatcher dispatcher
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype = detail::get_target_dtype(input1, input2);

        return dispatcher(
            at::autocast::cached_cast(target_dtype, input1),
            at::autocast::cached_cast(target_dtype, input2)
        );
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* schema_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            return forward(input1, input2, [schema_name](
                const at::Tensor& x1,
                const at::Tensor& x2
            ) {
                return c10::Dispatcher::singleton()
                    .findSchemaOrThrow(schema_name, "")
                    .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
                    .call(x1, x2);
            });
        });
    }
};

#define REGISTER_AUTOCAST_BINARY(module, name) \
    ::torchscience::autocast::AutocastBinaryOperator::register_all( \
        module, #name, "torchscience::" #name)

// ============================================================================
// AutocastTernaryOperator
// ============================================================================

struct AutocastTernaryOperator {
    template<typename Dispatcher>
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        Dispatcher dispatcher
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype = detail::get_target_dtype(input1, input2, input3);

        return dispatcher(
            at::autocast::cached_cast(target_dtype, input1),
            at::autocast::cached_cast(target_dtype, input2),
            at::autocast::cached_cast(target_dtype, input3)
        );
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* schema_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            return forward(input1, input2, input3, [schema_name](
                const at::Tensor& x1,
                const at::Tensor& x2,
                const at::Tensor& x3
            ) {
                return c10::Dispatcher::singleton()
                    .findSchemaOrThrow(schema_name, "")
                    .typed<at::Tensor(
                        const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&
                    )>()
                    .call(x1, x2, x3);
            });
        });
    }
};

#define REGISTER_AUTOCAST_TERNARY(module, name) \
    ::torchscience::autocast::AutocastTernaryOperator::register_all( \
        module, #name, "torchscience::" #name)

// ============================================================================
// AutocastQuaternaryOperator
// ============================================================================

struct AutocastQuaternaryOperator {
    template<typename Dispatcher>
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4,
        Dispatcher dispatcher
    ) {
        c10::impl::ExcludeDispatchKeyGuard exclude_autocast(
            c10::DispatchKey::Autocast
        );

        at::ScalarType target_dtype =
            detail::get_target_dtype(input1, input2, input3, input4);

        return dispatcher(
            at::autocast::cached_cast(target_dtype, input1),
            at::autocast::cached_cast(target_dtype, input2),
            at::autocast::cached_cast(target_dtype, input3),
            at::autocast::cached_cast(target_dtype, input4)
        );
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* schema_name
    ) {
        module.impl(name, [schema_name](
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            return forward(input1, input2, input3, input4, [schema_name](
                const at::Tensor& x1,
                const at::Tensor& x2,
                const at::Tensor& x3,
                const at::Tensor& x4
            ) {
                return c10::Dispatcher::singleton()
                    .findSchemaOrThrow(schema_name, "")
                    .typed<at::Tensor(
                        const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&
                    )>()
                    .call(x1, x2, x3, x4);
            });
        });
    }
};

#define REGISTER_AUTOCAST_QUATERNARY(module, name) \
    ::torchscience::autocast::AutocastQuaternaryOperator::register_all( \
        module, #name, "torchscience::" #name)

}  // namespace torchscience::autocast
