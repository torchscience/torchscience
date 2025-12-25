#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

/**
 * Autograd Function for Cook-Torrance BRDF.
 */
class CookTorrance
    : public torch::autograd::Function<CookTorrance> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& roughness,
        const at::Tensor& f0
    ) {
        bool normal_requires_grad = normal.requires_grad() && at::isFloatingType(normal.scalar_type());
        bool view_requires_grad = view.requires_grad() && at::isFloatingType(view.scalar_type());
        bool light_requires_grad = light.requires_grad() && at::isFloatingType(light.scalar_type());
        bool roughness_requires_grad = roughness.requires_grad() && at::isFloatingType(roughness.scalar_type());
        bool f0_requires_grad = f0.requires_grad() && at::isFloatingType(f0.scalar_type());

        ctx->saved_data["normal_requires_grad"] = normal_requires_grad;
        ctx->saved_data["view_requires_grad"] = view_requires_grad;
        ctx->saved_data["light_requires_grad"] = light_requires_grad;
        ctx->saved_data["roughness_requires_grad"] = roughness_requires_grad;
        ctx->saved_data["f0_requires_grad"] = f0_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cook_torrance", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(normal, view, light, roughness, f0);

        // Save tensors for backward
        ctx->save_for_backward({normal, view, light, roughness, f0});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor normal = saved[0];
        at::Tensor view = saved[1];
        at::Tensor light = saved[2];
        at::Tensor roughness = saved[3];
        at::Tensor f0 = saved[4];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool roughness_requires_grad = ctx->saved_data["roughness_requires_grad"].toBool();
        bool f0_requires_grad = ctx->saved_data["f0_requires_grad"].toBool();

        if (!normal_requires_grad && !view_requires_grad && !light_requires_grad &&
            !roughness_requires_grad && !f0_requires_grad) {
            return {
                at::Tensor(),  // grad_normal
                at::Tensor(),  // grad_view
                at::Tensor(),  // grad_light
                at::Tensor(),  // grad_roughness
                at::Tensor()   // grad_f0
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_view, grad_light, grad_roughness, grad_f0] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::cook_torrance_backward", "")
                .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&
                )>()
                .call(grad_output, normal, view, light, roughness, f0);

        return {
            normal_requires_grad ? grad_normal : at::Tensor(),
            view_requires_grad ? grad_view : at::Tensor(),
            light_requires_grad ? grad_light : at::Tensor(),
            roughness_requires_grad ? grad_roughness : at::Tensor(),
            f0_requires_grad ? grad_f0 : at::Tensor()
        };
    }
};

inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    return CookTorrance::apply(normal, view, light, roughness, f0);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("cook_torrance", &torchscience::autograd::graphics::shading::cook_torrance);
}
