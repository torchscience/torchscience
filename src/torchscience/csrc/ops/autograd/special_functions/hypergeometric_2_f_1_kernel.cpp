#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>
#include <torch/autograd.h>
#include <torch/library.h>
#include <torch/types.h>

#include <utility>

namespace science {
namespace ops {
namespace {

class Hypergeometric2F1Function : public torch::autograd::Function<Hypergeometric2F1Function> {
public:
    static torch::autograd::variable_list
    forward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::Variable& a,
        const torch::autograd::Variable& b,
        const torch::autograd::Variable& c,
        const torch::autograd::Variable& z
    ) {

        at::AutoDispatchBelowADInplaceOrView g;

        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "torchscience::hypergeometric_2_f_1",
            ""
        ).typed<at::Tensor(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&
        )>();

        auto output = op.call(a, b, c, z);

        context->save_for_backward({a, b, c, z, output});

        return {output};
    }

    static torch::autograd::variable_list
    backward(
        const torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_output
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();

        const at::Tensor &a = saved[0];
        const at::Tensor &b = saved[1];
        const at::Tensor &c = saved[2];
        const at::Tensor &z = saved[3];

        const at::Tensor &result = saved[4];

        const at::Tensor &grad_out = gradient_output[0];

        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "torchscience::_hypergeometric_2_f_1_backward",
            ""
        ).typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&
        )>();

        auto gradients = op.call(grad_out, a, b, c, z, result);

        return {
            std::get<0>(gradients),
            std::get<1>(gradients),
            std::get<2>(gradients),
            std::get<3>(gradients)
        };
    }
};

at::Tensor
hypergeometric_2_f_1_autograd(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    return Hypergeometric2F1Function::apply(a, b, c, z)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            hypergeometric_2_f_1_autograd
        )
    );
}

}  // namespace ops
}  // namespace science
