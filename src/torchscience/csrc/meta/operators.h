#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::meta {

// ============================================================================
// Meta operators compute output shapes only (no kernel dispatch)
// Used for shape inference in the Meta backend
// ============================================================================

// ============================================================================
// MetaUnaryOperator
// ============================================================================

struct MetaUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;

        return at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        at::Tensor grad_input;

        return at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        if (!grad_grad_input.defined()) {
            return {};
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input)
            .add_const_input(grad_grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1)};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_META_UNARY(module, name) \
    ::torchscience::meta::MetaUnaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// MetaBinaryOperator
// ============================================================================

struct MetaBinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor output;

        return at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1)};
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input1,
        const at::Tensor& grad_grad_input2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        const bool has_gg1 = grad_grad_input1.defined();
        const bool has_gg2 = grad_grad_input2.defined();

        if (!has_gg1 && !has_gg2) {
            return {};
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        at::Tensor gg1_input = has_gg1 ? grad_grad_input1 : at::zeros_like(grad_output);
        at::Tensor gg2_input = has_gg2 ? grad_grad_input2 : at::zeros_like(grad_output);

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(gg1_input)
            .add_const_input(gg2_input)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1), iter.output(2)};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_META_BINARY(module, name) \
    ::torchscience::meta::MetaBinaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// MetaTernaryOperator
// ============================================================================

struct MetaTernaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        at::Tensor output;

        return at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        at::Tensor grad_input1;
        at::Tensor grad_input2;
        at::Tensor grad_input3;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_output(grad_input3)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1), iter.output(2)};
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    backward_backward(
        const at::Tensor& grad_grad_input1,
        const at::Tensor& grad_grad_input2,
        const at::Tensor& grad_grad_input3,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        const bool has_gg1 = grad_grad_input1.defined();
        const bool has_gg2 = grad_grad_input2.defined();
        const bool has_gg3 = grad_grad_input3.defined();

        if (!has_gg1 && !has_gg2 && !has_gg3) {
            return {};
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input1;
        at::Tensor grad_input2;
        at::Tensor grad_input3;

        at::Tensor gg1_input = has_gg1 ? grad_grad_input1 : at::zeros_like(grad_output);
        at::Tensor gg2_input = has_gg2 ? grad_grad_input2 : at::zeros_like(grad_output);
        at::Tensor gg3_input = has_gg3 ? grad_grad_input3 : at::zeros_like(grad_output);

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_output(grad_input3)
            .add_const_input(gg1_input)
            .add_const_input(gg2_input)
            .add_const_input(gg3_input)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1), iter.output(2), iter.output(3)};
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_META_TERNARY(module, name) \
    ::torchscience::meta::MetaTernaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// MetaQuaternaryOperator
// ============================================================================

struct MetaQuaternaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        at::Tensor output;

        return at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .add_const_input(input4)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build()
            .output();
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        at::Tensor grad_input1;
        at::Tensor grad_input2;
        at::Tensor grad_input3;
        at::Tensor grad_input4;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_output(grad_input3)
            .add_output(grad_input4)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .add_const_input(input4)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {iter.output(0), iter.output(1), iter.output(2), iter.output(3)};
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    backward_backward(
        const at::Tensor& grad_grad_input1,
        const at::Tensor& grad_grad_input2,
        const at::Tensor& grad_grad_input3,
        const at::Tensor& grad_grad_input4,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        const bool has_gg1 = grad_grad_input1.defined();
        const bool has_gg2 = grad_grad_input2.defined();
        const bool has_gg3 = grad_grad_input3.defined();
        const bool has_gg4 = grad_grad_input4.defined();

        if (!has_gg1 && !has_gg2 && !has_gg3 && !has_gg4) {
            return {};
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input1;
        at::Tensor grad_input2;
        at::Tensor grad_input3;
        at::Tensor grad_input4;

        at::Tensor gg1_input = has_gg1 ? grad_grad_input1 : at::zeros_like(grad_output);
        at::Tensor gg2_input = has_gg2 ? grad_grad_input2 : at::zeros_like(grad_output);
        at::Tensor gg3_input = has_gg3 ? grad_grad_input3 : at::zeros_like(grad_output);
        at::Tensor gg4_input = has_gg4 ? grad_grad_input4 : at::zeros_like(grad_output);

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_output(grad_input3)
            .add_output(grad_input4)
            .add_const_input(gg1_input)
            .add_const_input(gg2_input)
            .add_const_input(gg3_input)
            .add_const_input(gg4_input)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .add_const_input(input4)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        return {
            iter.output(0), iter.output(1), iter.output(2),
            iter.output(3), iter.output(4)
        };
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_META_QUATERNARY(module, name) \
    ::torchscience::meta::MetaQuaternaryOperator::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::meta
