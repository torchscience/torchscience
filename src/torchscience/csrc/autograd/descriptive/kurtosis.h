#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::descriptive {

/**
 * Backward function class for double-backward support.
 */
class KurtosisBackward
    : public torch::autograd::Function<KurtosisBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input});

        // Store dim as vector since IntArrayRef doesn't persist
        if (dim.has_value()) {
            std::vector<int64_t> dim_vec(dim->begin(), dim->end());
            context->saved_data["dim"] = dim_vec;
            context->saved_data["has_dim"] = true;
        } else {
            context->saved_data["has_dim"] = false;
        }

        context->saved_data["keepdim"] = keepdim;
        context->saved_data["fisher"] = fisher;
        context->saved_data["bias"] = bias;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::OptionalIntArrayRef dim_ref;
        std::vector<int64_t> dim_vec_local;
        if (dim.has_value()) {
            dim_vec_local = std::vector<int64_t>(dim->begin(), dim->end());
            dim_ref = dim_vec_local;
        }

        at::Tensor grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_output, input, dim_ref, keepdim, fisher, bias);

        return {grad_input};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];

        bool has_dim = context->saved_data["has_dim"].toBool();
        bool keepdim = context->saved_data["keepdim"].toBool();
        bool fisher = context->saved_data["fisher"].toBool();
        bool bias = context->saved_data["bias"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::OptionalIntArrayRef dim_ref;
        std::vector<int64_t> dim_vec;
        if (has_dim) {
            dim_vec = context->saved_data["dim"].toIntVector();
            dim_ref = dim_vec;
        }

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_input
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_keepdim
                at::Tensor(),  // grad_fisher
                at::Tensor(),  // grad_bias
                at::Tensor()   // grad_input_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(grad_grad_input, grad_output, input, dim_ref, keepdim, fisher, bias);

        return {
            grad_grad_output,
            new_grad_input,
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_keepdim
            at::Tensor(),  // grad_fisher
            at::Tensor(),  // grad_bias
            at::Tensor()   // grad_input_requires_grad
        };
    }
};

/**
 * Forward function class with autograd support.
 */
class Kurtosis
    : public torch::autograd::Function<Kurtosis> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        bool fisher,
        bool bias
    ) {
        context->save_for_backward({input});

        if (dim.has_value()) {
            std::vector<int64_t> dim_vec(dim->begin(), dim->end());
            context->saved_data["dim"] = dim_vec;
            context->saved_data["has_dim"] = true;
        } else {
            context->saved_data["has_dim"] = false;
        }

        context->saved_data["keepdim"] = keepdim;
        context->saved_data["fisher"] = fisher;
        context->saved_data["bias"] = bias;

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::OptionalIntArrayRef dim_ref;
        std::vector<int64_t> dim_vec_local;
        if (dim.has_value()) {
            dim_vec_local = std::vector<int64_t>(dim->begin(), dim->end());
            dim_ref = dim_vec_local;
        }

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kurtosis", "")
            .typed<at::Tensor(
                const at::Tensor&,
                at::OptionalIntArrayRef,
                bool,
                bool,
                bool
            )>()
            .call(input, dim_ref, keepdim, fisher, bias);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];

        at::Tensor grad_output = grad_outputs[0];

        bool has_dim = context->saved_data["has_dim"].toBool();
        bool keepdim = context->saved_data["keepdim"].toBool();
        bool fisher = context->saved_data["fisher"].toBool();
        bool bias = context->saved_data["bias"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {
                at::Tensor(),  // grad_input
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_keepdim
                at::Tensor(),  // grad_fisher
                at::Tensor()   // grad_bias
            };
        }

        at::OptionalIntArrayRef dim_ref;
        std::vector<int64_t> dim_vec;
        if (has_dim) {
            dim_vec = context->saved_data["dim"].toIntVector();
            dim_ref = dim_vec;
        }

        std::vector<at::Tensor> gradients = KurtosisBackward::apply(
            grad_output,
            input,
            dim_ref,
            keepdim,
            fisher,
            bias,
            input_requires_grad
        );

        return {
            gradients[0],  // grad_input
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_keepdim
            at::Tensor(),  // grad_fisher
            at::Tensor()   // grad_bias
        };
    }
};

inline at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    return Kurtosis::apply(input, dim, keepdim, fisher, bias);
}

}  // namespace torchscience::autograd::descriptive

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "kurtosis",
        &torchscience::autograd::descriptive::kurtosis
    );
}
