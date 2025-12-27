#pragma once

#include <torch/extension.h>

namespace torchscience::autograd {

class GammaBackward : public torch::autograd::Function<GammaBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        bool input_requires_grad
    ) {
        ctx->save_for_backward({grad_output, input});
        ctx->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gamma_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>();
        return {op.call(grad_output, input)};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        bool input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();

        if (!grad_outputs[0].defined() || !input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gamma_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
        auto [gg_out, new_grad] = op.call(grad_outputs[0], saved[0], saved[1]);
        return {gg_out, new_grad, at::Tensor()};
    }
};

class GammaForward : public torch::autograd::Function<GammaForward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input
    ) {
        ctx->save_for_backward({input});
        ctx->saved_data["input_requires_grad"] =
            input.requires_grad() &&
            (at::isFloatingType(input.scalar_type()) || at::isComplexType(input.scalar_type()));

        at::AutoDispatchBelowAutograd guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gamma", "")
            .typed<at::Tensor(const at::Tensor&)>();
        return op.call(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        bool input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();
        auto grads = GammaBackward::apply(grad_outputs[0], saved[0], input_requires_grad);
        return {input_requires_grad ? grads[0] : at::Tensor()};
    }
};

inline at::Tensor gamma_autograd(const at::Tensor& input) {
    return GammaForward::apply(input);
}

}  // namespace torchscience::autograd

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gamma", torchscience::autograd::gamma_autograd);
}
