#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd {

// =============================================================================
// AutogradReductionOperator - Autograd support for reduction operators
// =============================================================================

// Usage: Define a traits struct and dispatcher functions, then use this template
// to create the autograd-enabled version.

template<typename Dispatcher>
struct AutogradReductionOperator {
    class Backward : public torch::autograd::Function<Backward> {
    public:
        template<typename... Args>
        static std::vector<at::Tensor> forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& grad_output,
            const at::Tensor& input,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            bool input_requires_grad,
            Args... args
        ) {
            context->save_for_backward({grad_output, input});

            if (dim.has_value()) {
                std::vector<int64_t> dim_vec(dim->begin(), dim->end());
                context->saved_data["dim"] = dim_vec;
                context->saved_data["has_dim"] = true;
            } else {
                context->saved_data["has_dim"] = false;
            }

            context->saved_data["keepdim"] = keepdim;
            context->saved_data["input_requires_grad"] = input_requires_grad;

            at::AutoDispatchBelowAutograd guard;

            at::OptionalIntArrayRef dim_ref;
            std::vector<int64_t> dim_vec_local;
            if (dim.has_value()) {
                dim_vec_local = std::vector<int64_t>(dim->begin(), dim->end());
                dim_ref = dim_vec_local;
            }

            at::Tensor grad_input = Dispatcher::dispatch_backward(
                grad_output, input, dim_ref, keepdim, args...
            );

            return {grad_input};
        }

        static std::vector<at::Tensor> backward(
            torch::autograd::AutogradContext* context,
            const std::vector<at::Tensor>& grad_outputs
        ) {
            const auto saved = context->get_saved_variables();
            at::Tensor grad_output = saved[0];
            at::Tensor input = saved[1];

            bool has_dim = context->saved_data["has_dim"].toBool();
            bool keepdim = context->saved_data["keepdim"].toBool();
            bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

            at::OptionalIntArrayRef dim_ref;
            std::vector<int64_t> dim_vec;
            if (has_dim) {
                dim_vec = context->saved_data["dim"].toIntVector();
                dim_ref = dim_vec;
            }

            at::Tensor grad_grad_input = grad_outputs[0];

            if (!grad_grad_input.defined() || !input_requires_grad) {
                // Return empty gradients for all inputs
                return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
            }

            at::AutoDispatchBelowAutograd guard;

            auto [grad_grad_output, new_grad_input] = Dispatcher::dispatch_backward_backward(
                grad_grad_input, grad_output, input, dim_ref, keepdim
            );

            return {grad_grad_output, new_grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
        }
    };

    class Forward : public torch::autograd::Function<Forward> {
    public:
        template<typename... Args>
        static at::Tensor forward(
            torch::autograd::AutogradContext* context,
            const at::Tensor& input,
            at::OptionalIntArrayRef dim,
            bool keepdim,
            Args... args
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

            return Dispatcher::dispatch_forward(input, dim_ref, keepdim, args...);
        }

        template<typename... Args>
        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* context,
            const torch::autograd::variable_list& grad_outputs,
            Args... args
        ) {
            const auto saved = context->get_saved_variables();
            at::Tensor input = saved[0];
            at::Tensor grad_output = grad_outputs[0];

            bool has_dim = context->saved_data["has_dim"].toBool();
            bool keepdim = context->saved_data["keepdim"].toBool();
            bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

            if (!input_requires_grad) {
                return {at::Tensor(), at::Tensor(), at::Tensor()};
            }

            at::OptionalIntArrayRef dim_ref;
            std::vector<int64_t> dim_vec;
            if (has_dim) {
                dim_vec = context->saved_data["dim"].toIntVector();
                dim_ref = dim_vec;
            }

            std::vector<at::Tensor> gradients = Backward::apply(
                grad_output, input, dim_ref, keepdim, input_requires_grad, args...
            );

            return {gradients[0], at::Tensor(), at::Tensor()};
        }
    };
};

}  // namespace torchscience::autograd
