#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::integral_transform {

/**
 * Backward function class for inverse Hilbert transform double-backward support.
 */
class InverseHilbertTransformBackward
    : public torch::autograd::Function<InverseHilbertTransformBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t n,
        int64_t dim,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input});
        context->saved_data["n"] = n;
        context->saved_data["dim"] = dim;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
            .call(grad_output, input, n, dim);

        return {grad_input};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];

        int64_t n = context->saved_data["n"].toInt();
        int64_t dim = context->saved_data["dim"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor()
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hilbert_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t
            )>()
            .call(grad_grad_input, grad_output, input, n, dim);

        return {
            grad_grad_output,
            new_grad_input,
            at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

/**
 * Forward function class for inverse Hilbert transform with autograd support.
 */
class InverseHilbertTransform
    : public torch::autograd::Function<InverseHilbertTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        int64_t n,
        int64_t dim
    ) {
        context->save_for_backward({input});
        context->saved_data["n"] = n;
        context->saved_data["dim"] = dim;

        bool input_requires_grad = input.requires_grad() &&
            (at::isFloatingType(input.scalar_type()) || at::isComplexType(input.scalar_type()));
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hilbert_transform", "")
            .typed<at::Tensor(const at::Tensor&, int64_t, int64_t)>()
            .call(input, n, dim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor grad_output = grad_outputs[0];

        int64_t n = context->saved_data["n"].toInt();
        int64_t dim = context->saved_data["dim"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor()
            };
        }

        std::vector<at::Tensor> gradients = InverseHilbertTransformBackward::apply(
            grad_output,
            input,
            n,
            dim,
            input_requires_grad
        );

        return {
            gradients[0],
            at::Tensor(),
            at::Tensor()
        };
    }
};

inline at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n,
    int64_t dim
) {
    return InverseHilbertTransform::apply(input, n, dim);
}

}  // namespace torchscience::autograd::integral_transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::autograd::integral_transform::inverse_hilbert_transform
    );
}
