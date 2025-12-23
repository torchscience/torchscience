#pragma once

#include <tuple>

#include <torch/extension.h>

namespace torchscience::autograd {

// ============================================================================
// AutogradUnaryOperator - Template for unary operators with custom autograd
// ============================================================================

template<typename ImplTraits>
struct AutogradUnaryOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& grad_output,
            const at::Tensor& input,
            bool input_requires_grad
        ) {
            context->save_for_backward({grad_output, input});
            context->saved_data["input_requires_grad"] = input_requires_grad;

            at::AutoDispatchBelowAutograd guard;

            at::Tensor grad_input = ImplTraits::dispatch_backward(grad_output, input);

            return {grad_input};
        }

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* context,
            const std::vector<at::Tensor>& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            const bool input_requires_grad =
                context->saved_data["input_requires_grad"].toBool();

            const bool grad_input_defined = grad_outputs[0].defined();

            if (!(grad_input_defined && input_requires_grad)) {
                return {at::Tensor(), at::Tensor(), at::Tensor()};
            }

            at::AutoDispatchBelowAutograd guard;

            at::Tensor gg_input;
            if (grad_input_defined && input_requires_grad) {
                gg_input = grad_outputs[0];
            }

            auto [grad_grad_output, grad_input] =
                ImplTraits::dispatch_backward_backward(gg_input, saved[0], saved[1]);

            return {grad_grad_output, grad_input, at::Tensor()};
        }
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& input
        ) {
            context->save_for_backward({input});

            const bool condition = isFloatingType(input.scalar_type()) ||
                                   isComplexType(input.scalar_type());

            context->saved_data["input_requires_grad"] =
                input.requires_grad() && condition;

            at::AutoDispatchBelowAutograd guard;

            return ImplTraits::dispatch_forward(input);
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* context,
            const torch::autograd::variable_list& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            at::Tensor input = saved[0];
            at::Tensor grad_output = grad_outputs[0];

            bool input_requires_grad =
                context->saved_data["input_requires_grad"].toBool();

            std::vector<at::Tensor> gradients = Backward::apply(
                grad_output, input, input_requires_grad
            );

            at::Tensor grad_input;
            if (input_requires_grad) {
                grad_input = gradients[0];
            }

            return {grad_input};
        }
    };

    static at::Tensor apply(const at::Tensor& input) {
        return Forward::apply(input);
    }

    static void register_all(torch::Library& module, const char* name) {
        module.impl(name, &apply);
    }
};

#define REGISTER_AUTOGRAD_UNARY(module, name, Impl) \
    ::torchscience::autograd::AutogradUnaryOperator<Impl>::register_all(module, #name)

// ============================================================================
// AutogradBinaryOperator - Template for binary operators with custom autograd
// ============================================================================

template<typename ImplTraits>
struct AutogradBinaryOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            bool input1_requires_grad
        ) {
            context->save_for_backward({grad_output, input1, input2});
            context->saved_data["input1_requires_grad"] = input1_requires_grad;

            at::AutoDispatchBelowAutograd guard;

            auto [grad_input1, grad_input2] =
                ImplTraits::dispatch_backward(grad_output, input1, input2);

            return {grad_input1, grad_input2};
        }

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* context,
            const std::vector<at::Tensor>& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            const bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            const bool grad_input1_defined = grad_outputs[0].defined();
            const bool grad_input2_defined = grad_outputs[1].defined();

            if (!(grad_input1_defined && input1_requires_grad) &&
                !grad_input2_defined) {
                return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
            }

            at::AutoDispatchBelowAutograd guard;

            at::Tensor gg_input1;
            if (grad_input1_defined && input1_requires_grad) {
                gg_input1 = grad_outputs[0];
            }

            at::Tensor gg_input2;
            if (grad_input2_defined) {
                gg_input2 = grad_outputs[1];
            }

            auto [grad_grad_output, grad_input1, grad_input2] =
                ImplTraits::dispatch_backward_backward(
                    gg_input1, gg_input2, saved[0], saved[1], saved[2]
                );

            return {grad_grad_output, grad_input1, grad_input2, at::Tensor()};
        }
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& input1,
            const at::Tensor& input2
        ) {
            context->save_for_backward({input1, input2});

            const bool condition = isFloatingType(input1.scalar_type()) ||
                                   isComplexType(input1.scalar_type());

            context->saved_data["input1_requires_grad"] =
                input1.requires_grad() && condition;

            at::AutoDispatchBelowAutograd guard;

            return ImplTraits::dispatch_forward(input1, input2);
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* context,
            const torch::autograd::variable_list& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            at::Tensor input1 = saved[0];
            at::Tensor input2 = saved[1];
            at::Tensor grad_output = grad_outputs[0];

            bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            std::vector<at::Tensor> gradients = Backward::apply(
                grad_output, input1, input2, input1_requires_grad
            );

            at::Tensor grad_input1;
            if (input1_requires_grad) {
                grad_input1 = gradients[0];
            }

            return {grad_input1, gradients[1]};
        }
    };

    static at::Tensor apply(const at::Tensor& input1, const at::Tensor& input2) {
        return Forward::apply(input1, input2);
    }

    static void register_all(torch::Library& module, const char* name) {
        module.impl(name, &apply);
    }
};

#define REGISTER_AUTOGRAD_BINARY(module, name, Impl) \
    ::torchscience::autograd::AutogradBinaryOperator<Impl>::register_all(module, #name)

// ============================================================================
// AutogradTernaryOperator - Template for ternary operators with custom autograd
// ============================================================================

template<typename ImplTraits>
struct AutogradTernaryOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            bool input1_requires_grad
        ) {
            context->save_for_backward({grad_output, input1, input2, input3});
            context->saved_data["input1_requires_grad"] = input1_requires_grad;

            at::AutoDispatchBelowAutograd guard;

            auto [grad_input1, grad_input2, grad_input3] =
                ImplTraits::dispatch_backward(grad_output, input1, input2, input3);

            return {grad_input1, grad_input2, grad_input3};
        }

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* context,
            const std::vector<at::Tensor>& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            const bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            const bool grad_input1_defined = grad_outputs[0].defined();
            const bool grad_input2_defined = grad_outputs[1].defined();
            const bool grad_input3_defined = grad_outputs[2].defined();

            if (!(grad_input1_defined && input1_requires_grad) &&
                !grad_input2_defined && !grad_input3_defined) {
                return {
                    at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor()
                };
            }

            at::AutoDispatchBelowAutograd guard;

            at::Tensor gg_input1;
            if (grad_input1_defined && input1_requires_grad) {
                gg_input1 = grad_outputs[0];
            }

            at::Tensor gg_input2;
            if (grad_input2_defined) {
                gg_input2 = grad_outputs[1];
            }

            at::Tensor gg_input3;
            if (grad_input3_defined) {
                gg_input3 = grad_outputs[2];
            }

            auto [grad_grad_output, grad_input1, grad_input2, grad_input3] =
                ImplTraits::dispatch_backward_backward(
                    gg_input1, gg_input2, gg_input3,
                    saved[0], saved[1], saved[2], saved[3]
                );

            return {
                grad_grad_output, grad_input1, grad_input2, grad_input3,
                at::Tensor()
            };
        }
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3
        ) {
            context->save_for_backward({input1, input2, input3});

            const bool condition = isFloatingType(input1.scalar_type()) ||
                                   isComplexType(input1.scalar_type());

            context->saved_data["input1_requires_grad"] =
                input1.requires_grad() && condition;

            at::AutoDispatchBelowAutograd guard;

            return ImplTraits::dispatch_forward(input1, input2, input3);
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* context,
            const torch::autograd::variable_list& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            at::Tensor input1 = saved[0];
            at::Tensor input2 = saved[1];
            at::Tensor input3 = saved[2];
            at::Tensor grad_output = grad_outputs[0];

            bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            std::vector<at::Tensor> gradients = Backward::apply(
                grad_output, input1, input2, input3, input1_requires_grad
            );

            at::Tensor grad_input1;
            if (input1_requires_grad) {
                grad_input1 = gradients[0];
            }

            return {grad_input1, gradients[1], gradients[2]};
        }
    };

    static at::Tensor apply(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3
    ) {
        return Forward::apply(input1, input2, input3);
    }

    static void register_all(torch::Library& module, const char* name) {
        module.impl(name, &apply);
    }
};

#define REGISTER_AUTOGRAD_TERNARY(module, name, Impl) \
    ::torchscience::autograd::AutogradTernaryOperator<Impl>::register_all(module, #name)

// ============================================================================
// AutogradQuaternaryOperator - Template for quaternary operators with custom autograd
// ============================================================================

template<typename ImplTraits>
struct AutogradQuaternaryOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& grad_output,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4,
            bool input1_requires_grad
        ) {
            context->save_for_backward({grad_output, input1, input2, input3, input4});
            context->saved_data["input1_requires_grad"] = input1_requires_grad;

            at::AutoDispatchBelowAutograd guard;

            auto [grad_input1, grad_input2, grad_input3, grad_input4] =
                ImplTraits::dispatch_backward(
                    grad_output, input1, input2, input3, input4
                );

            return {grad_input1, grad_input2, grad_input3, grad_input4};
        }

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* context,
            const std::vector<at::Tensor>& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            const bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            const bool grad_input1_defined = grad_outputs[0].defined();
            const bool grad_input2_defined = grad_outputs[1].defined();
            const bool grad_input3_defined = grad_outputs[2].defined();
            const bool grad_input4_defined = grad_outputs[3].defined();

            if (!(grad_input1_defined && input1_requires_grad) &&
                !grad_input2_defined && !grad_input3_defined &&
                !grad_input4_defined) {
                return {
                    at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()
                };
            }

            at::AutoDispatchBelowAutograd guard;

            at::Tensor gg_input1;
            if (grad_input1_defined && input1_requires_grad) {
                gg_input1 = grad_outputs[0];
            }

            at::Tensor gg_input2;
            if (grad_input2_defined) {
                gg_input2 = grad_outputs[1];
            }

            at::Tensor gg_input3;
            if (grad_input3_defined) {
                gg_input3 = grad_outputs[2];
            }

            at::Tensor gg_input4;
            if (grad_input4_defined) {
                gg_input4 = grad_outputs[3];
            }

            auto [grad_grad_output, grad_input1, grad_input2, grad_input3, grad_input4] =
                ImplTraits::dispatch_backward_backward(
                    gg_input1, gg_input2, gg_input3, gg_input4,
                    saved[0], saved[1], saved[2], saved[3], saved[4]
                );

            return {
                grad_grad_output, grad_input1, grad_input2, grad_input3, grad_input4,
                at::Tensor()
            };
        }
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& input1,
            const at::Tensor& input2,
            const at::Tensor& input3,
            const at::Tensor& input4
        ) {
            context->save_for_backward({input1, input2, input3, input4});

            const bool condition = isFloatingType(input1.scalar_type()) ||
                                   isComplexType(input1.scalar_type());

            context->saved_data["input1_requires_grad"] =
                input1.requires_grad() && condition;

            at::AutoDispatchBelowAutograd guard;

            return ImplTraits::dispatch_forward(input1, input2, input3, input4);
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* context,
            const torch::autograd::variable_list& grad_outputs
        ) {
            const torch::autograd::variable_list saved = context->get_saved_variables();
            at::Tensor input1 = saved[0];
            at::Tensor input2 = saved[1];
            at::Tensor input3 = saved[2];
            at::Tensor input4 = saved[3];
            at::Tensor grad_output = grad_outputs[0];

            bool input1_requires_grad =
                context->saved_data["input1_requires_grad"].toBool();

            std::vector<at::Tensor> gradients = Backward::apply(
                grad_output, input1, input2, input3, input4, input1_requires_grad
            );

            at::Tensor grad_input1;
            if (input1_requires_grad) {
                grad_input1 = gradients[0];
            }

            return {grad_input1, gradients[1], gradients[2], gradients[3]};
        }
    };

    static at::Tensor apply(
        const at::Tensor& input1,
        const at::Tensor& input2,
        const at::Tensor& input3,
        const at::Tensor& input4
    ) {
        return Forward::apply(input1, input2, input3, input4);
    }

    static void register_all(torch::Library& module, const char* name) {
        module.impl(name, &apply);
    }
};

#define REGISTER_AUTOGRAD_QUATERNARY(module, name, Impl) \
    ::torchscience::autograd::AutogradQuaternaryOperator<Impl>::register_all(module, #name)

}  // namespace torchscience::autograd
