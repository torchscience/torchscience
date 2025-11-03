#include "../example.h"

#include <torch/autograd.h>
#include <torch/types.h>

#include <utility>

namespace science {
namespace ops {
namespace {
class ExampleFunction : public torch::autograd::Function<ExampleFunction> {
public:
    static torch::autograd::variable_list
    forward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::Variable& input,
        const c10::SymInt& foo,
        const c10::SymInt& bar,
        const c10::SymInt& baz,
    ) {
        at::AutoDispatchBelowADInplaceOrView g;

        auto output = example_symint(
            input,
            foo,
            bar,
            baz,
        );

        context->save_for_backward({input});

        context->saved_data["foo"] = foo;
        context->saved_data["bar"] = bar;
        context->saved_data["baz"] = baz;

        return {
            output,
        };
    }

    static torch::autograd::variable_list
    backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_output
    ) {
        auto saved = context->get_saved_variables();

        auto input = saved[0];

        auto foo = context->saved_data["foo"].toSymInt();
        auto bar = context->saved_data["bar"].toSymInt();
        auto baz = context->saved_data["baz"].toSymInt();

        auto gradients = detail::_example_backward_symint(
            gradient_output[0],
            input,
            foo,
            bar,
            baz,
        );

        auto gradient_input = std::get<0>(gradients);

        return {
            gradient_input,
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
        };
    }
};

class ExampleBackwardFunction : public torch::autograd::Function<ExampleBackwardFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::Variable& grad,
        const torch::autograd::Variable& input,
        const torch::autograd::Variable& weight,
        const torch::autograd::Variable& offset,
        const torch::autograd::Variable& mask,
        const torch::autograd::Variable& bias,
        c10::SymInt stride_h,
        c10::SymInt stride_w,
        c10::SymInt pad_h,
        c10::SymInt pad_w,
        c10::SymInt dilation_h,
        c10::SymInt dilation_w,
        c10::SymInt groups,
        c10::SymInt offset_groups,
        bool use_mask) {
      at::AutoDispatchBelowADInplaceOrView g;
      auto result = detail::_example_backward_symint(
          grad,
          input,
          weight,
          offset,
          mask,
          bias,
          std::move(stride_h),
          std::move(stride_w),
          std::move(pad_h),
          std::move(pad_w),
          std::move(dilation_h),
          std::move(dilation_w),
          std::move(groups),
          std::move(offset_groups),
          use_mask);

      auto gradient_input = std::get<0>(result);
      auto grad_weight = std::get<1>(result);
      auto grad_offset = std::get<2>(result);
      auto grad_mask = std::get<3>(result);
      auto grad_bias = std::get<4>(result);

      return {
          gradient_input,
          grad_weight,
          grad_offset,
          grad_mask,
          grad_bias,
      };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_output
    ) {
      TORCH_CHECK(0, "double backwards on example not supported");
    }
};

at::Tensor example_autograd(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask) {
  return ExampleFunction::apply(
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask)[0];
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
example_backward_autograd(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    c10::SymInt stride_h,
    c10::SymInt stride_w,
    c10::SymInt pad_h,
    c10::SymInt pad_w,
    c10::SymInt dilation_h,
    c10::SymInt dilation_w,
    c10::SymInt groups,
    c10::SymInt offset_groups,
    bool use_mask) {
  auto result = ExampleBackwardFunction::apply(
      grad,
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask);

  return std::make_tuple(result[0], result[1], result[2], result[3], result[4]);
}
} // namespace

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
  module.impl(
      TORCH_SELECTIVE_NAME("torchscience::example"),
      TORCH_FN(example_autograd)
  );

  module.impl(
      TORCH_SELECTIVE_NAME("torchscience::_example_backward"),
      TORCH_FN(example_backward_autograd)
  );
}
} // namespace ops
} // namespace science
