#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKey.h>
#include <torch/autograd.h>
#include <torch/library.h>
#include <torch/types.h>

#include <utility>

namespace science {
namespace ops {
namespace {

// Custom autograd function for the example operator
class ExampleFunction : public torch::autograd::Function<ExampleFunction> {
  public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* context,
                                                  const torch::autograd::Variable& input,
                                                  const at::Scalar& x) {
        at::AutoDispatchBelowADInplaceOrView g;

        // Call the actual example operator through the dispatcher
        static auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("torchscience::example", "")
                             .typed<at::Tensor(const at::Tensor&, const at::Scalar&)>();

        auto output = op.call(input, x);

        context->save_for_backward({input});
        context->saved_data["x"] = x;

        return {output};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_output) {
        auto saved = context->get_saved_variables();
        auto input = saved[0];

        auto x = context->saved_data["x"].toScalar();

        // Call the backward operator through the dispatcher
        static auto op =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::_example_backward", "")
                .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Scalar&)>();

        auto gradient_input = op.call(gradient_output[0], input, x);

        return {gradient_input, torch::autograd::Variable()};
    }
};

at::Tensor example_autograd(const at::Tensor& input, const at::Scalar& x) {
    return ExampleFunction::apply(input, x)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(TORCH_SELECTIVE_NAME("torchscience::example"), TORCH_FN(example_autograd));
}

}  // namespace ops
}  // namespace science
