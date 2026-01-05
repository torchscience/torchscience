#pragma once

#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class CrossEntropyBackward : public torch::autograd::Function<CrossEntropyBackward> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->save_for_backward({grad_output, p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>();

        auto result = op.redispatch(
            c10::DispatchKeySet({c10::DispatchKey::CPU, c10::DispatchKey::CUDA}),
            grad_output, p, q, dim, input_type, reduction, base
        );

        return {std::get<0>(result), std::get<1>(result)};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto grad_output = saved[0];
        auto p = saved[1];
        auto q = saved[2];

        auto dim = ctx->saved_data["dim"].toInt();
        auto input_type = ctx->saved_data["input_type"].toStringRef();
        auto reduction = ctx->saved_data["reduction"].toStringRef();
        auto base = ctx->saved_data["base"].toOptional<double>();

        auto gg_p = grad_outputs[0];
        auto gg_q = grad_outputs[1];

        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>();

        auto result = op.redispatch(
            c10::DispatchKeySet({c10::DispatchKey::CPU, c10::DispatchKey::CUDA}),
            gg_p, gg_q, grad_output, p, q, dim, input_type, reduction, base
        );

        return {
            std::get<0>(result),  // grad_grad_output
            std::get<1>(result),  // grad_p
            std::get<2>(result),  // grad_q
            at::Tensor(),         // dim
            at::Tensor(),         // input_type
            at::Tensor(),         // reduction
            at::Tensor()          // base
        };
    }
};

class CrossEntropyFunction : public torch::autograd::Function<CrossEntropyFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->save_for_backward({p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>();

        return op.redispatch(
            c10::DispatchKeySet({c10::DispatchKey::CPU, c10::DispatchKey::CUDA}),
            p, q, dim, input_type, reduction, base
        );
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto p = saved[0];
        auto q = saved[1];

        auto dim = ctx->saved_data["dim"].toInt();
        auto input_type = ctx->saved_data["input_type"].toStringRef();
        auto reduction = ctx->saved_data["reduction"].toStringRef();
        auto base = ctx->saved_data["base"].toOptional<double>();

        auto grad_output = grad_outputs[0];

        auto grads = CrossEntropyBackward::apply(
            grad_output, p, q, dim, input_type, reduction, base
        );

        return {
            grads[0],     // grad_p
            grads[1],     // grad_q
            at::Tensor(), // dim
            at::Tensor(), // input_type
            at::Tensor(), // reduction
            at::Tensor()  // base
        };
    }
};

inline at::Tensor cross_entropy(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return CrossEntropyFunction::apply(p, q, dim, input_type, reduction, base);
}

inline std::tuple<at::Tensor, at::Tensor> cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    auto result = CrossEntropyBackward::apply(
        grad_output, p, q, dim, input_type, reduction, base
    );
    return std::make_tuple(result[0], result[1]);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("cross_entropy", &torchscience::autograd::information_theory::cross_entropy);
    m.impl("cross_entropy_backward", &torchscience::autograd::information_theory::cross_entropy_backward);
}
