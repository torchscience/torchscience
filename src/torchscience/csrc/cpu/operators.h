#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::cpu {

// ============================================================================
// CPUUnaryOperator - Template for unary operators with first and second
// order derivatives
// ============================================================================

template<typename ImplTraits>
struct CPUUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_forward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t x) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x);
                });
            }
        );

        return iter.output();
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_backward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t g, scalar_t x) -> scalar_t {
                    return ImplTraits::template backward<scalar_t>(g, x);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        const bool has_gg = grad_grad_input.defined();

        if (!has_gg) {
            return std::make_tuple(at::Tensor(), at::Tensor());
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "unary_backward_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [has_gg](scalar_t gg, scalar_t g, scalar_t x)
                        -> std::tuple<scalar_t, scalar_t> {
                        return ImplTraits::template backward_backward<scalar_t>(
                            gg, g, x, has_gg
                        );
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
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

// Minimal macro for string concatenation only
#define REGISTER_CPU_UNARY(module, name, Impl) \
    ::torchscience::cpu::CPUUnaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// CPUBinaryOperator - Template for binary operators with first and second
// order derivatives
// ============================================================================

template<typename ImplTraits>
struct CPUBinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "binary_forward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t x1, scalar_t x2) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x1, x2);
                });
            }
        );

        return iter.output();
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "binary_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [](scalar_t g, scalar_t x1, scalar_t x2)
                        -> std::tuple<scalar_t, scalar_t> {
                        return ImplTraits::template backward<scalar_t>(g, x1, x2);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
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
            return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "binary_backward_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [has_gg1, has_gg2](
                        scalar_t gg1, scalar_t gg2, scalar_t g, scalar_t x1, scalar_t x2
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return ImplTraits::template backward_backward<scalar_t>(
                            gg1, gg2, g, x1, x2, has_gg1, has_gg2
                        );
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1), iter.output(2));
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

#define REGISTER_CPU_BINARY(module, name, Impl) \
    ::torchscience::cpu::CPUBinaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// CPUTernaryOperator - Template for ternary operators with first and second
// order derivatives
// ============================================================================

template<typename ImplTraits>
struct CPUTernaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "ternary_forward",
            [&]() {
                at::native::cpu_kernel(iter, [](scalar_t x1, scalar_t x2, scalar_t x3) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x1, x2, x3);
                });
            }
        );

        return iter.output();
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "ternary_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [](scalar_t g, scalar_t x1, scalar_t x2, scalar_t x3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return ImplTraits::template backward<scalar_t>(g, x1, x2, x3);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1), iter.output(2));
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> backward_backward(
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
            return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor());
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "ternary_backward_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [has_gg1, has_gg2, has_gg3](
                        scalar_t gg1, scalar_t gg2, scalar_t gg3, scalar_t g,
                        scalar_t x1, scalar_t x2, scalar_t x3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return ImplTraits::template backward_backward<scalar_t>(
                            gg1, gg2, gg3, g, x1, x2, x3, has_gg1, has_gg2, has_gg3
                        );
                    }
                );
            }
        );

        return std::make_tuple(
            iter.output(0), iter.output(1), iter.output(2), iter.output(3)
        );
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

#define REGISTER_CPU_TERNARY(module, name, Impl) \
    ::torchscience::cpu::CPUTernaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

// ============================================================================
// CPUQuaternaryOperator - Template for quaternary operators with first and
// second order derivatives
// ============================================================================

template<typename ImplTraits>
struct CPUQuaternaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .add_const_input(input3)
            .add_const_input(input4)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "quaternary_forward",
            [&]() {
                at::native::cpu_kernel(iter, [](
                    scalar_t x1, scalar_t x2, scalar_t x3, scalar_t x4
                ) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x1, x2, x3, x4);
                });
            }
        );

        return iter.output();
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "quaternary_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [](scalar_t g, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t x4)
                        -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return ImplTraits::template backward<scalar_t>(g, x1, x2, x3, x4);
                    }
                );
            }
        );

        return std::make_tuple(
            iter.output(0), iter.output(1), iter.output(2), iter.output(3)
        );
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
            return std::make_tuple(
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
            );
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

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "quaternary_backward_backward",
            [&]() {
                at::native::cpu_kernel_multiple_outputs(
                    iter,
                    [has_gg1, has_gg2, has_gg3, has_gg4](
                        scalar_t gg1, scalar_t gg2, scalar_t gg3, scalar_t gg4,
                        scalar_t g, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t x4
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> {
                        return ImplTraits::template backward_backward<scalar_t>(
                            gg1, gg2, gg3, gg4, g, x1, x2, x3, x4,
                            has_gg1, has_gg2, has_gg3, has_gg4
                        );
                    }
                );
            }
        );

        return std::make_tuple(
            iter.output(0), iter.output(1), iter.output(2),
            iter.output(3), iter.output(4)
        );
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

#define REGISTER_CPU_QUATERNARY(module, name, Impl) \
    ::torchscience::cpu::CPUQuaternaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::cpu
