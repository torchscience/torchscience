#pragma once

#include <string>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/Dispatcher.h>

namespace torchscience::core {

// ============================================================================
// Dispatch helpers for traits structs
// These reduce boilerplate by generating dispatch methods from operator names
// ============================================================================

// Helper to build schema name at compile time
inline std::string make_schema(const char* op_name, const char* suffix = "") {
    return std::string("torchscience::") + op_name + suffix;
}

// ============================================================================
// Unary dispatch helpers
// ============================================================================

template<const char* OpName>
struct UnaryDispatch {
    static at::Tensor forward(const at::Tensor& input) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName).c_str(), "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(input);
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward").c_str(), "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_output, input);
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_input, grad_output, input);
    }
};

// ============================================================================
// Binary dispatch helpers
// ============================================================================

template<const char* OpName>
struct BinaryDispatch {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName).c_str(), "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(input1, input2);
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, input1, input2);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_1,
        const at::Tensor& grad_grad_2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_1, grad_grad_2, grad_output, input1, input2);
    }
};

// ============================================================================
// Ternary dispatch helpers
// ============================================================================

template<const char* OpName>
struct TernaryDispatch {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName).c_str(), "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(input1, input2, input3);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, input1, input2, input3);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_1,
        const at::Tensor& grad_grad_2,
        const at::Tensor& grad_grad_3,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_1, grad_grad_2, grad_grad_3, grad_output, input1, input2, input3);
    }
};

// ============================================================================
// Quaternary dispatch helpers
// ============================================================================

template<const char* OpName>
struct QuaternaryDispatch {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName).c_str(), "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(input1, input2, input3, input4);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, input1, input2, input3, input4);
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_1,
        const at::Tensor& grad_grad_2,
        const at::Tensor& grad_grad_3,
        const at::Tensor& grad_grad_4,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow(make_schema(OpName, "_backward_backward").c_str(), "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_grad_1, grad_grad_2, grad_grad_3, grad_grad_4,
                  grad_output, input1, input2, input3, input4);
    }
};

}  // namespace torchscience::core

// ============================================================================
// Operator name declarations for template parameters
// Usage: DECLARE_OP_NAME(gamma) creates constexpr char gamma_op_name[]
// ============================================================================

#define DECLARE_OP_NAME(name) \
    inline constexpr char name##_op_name[] = #name
