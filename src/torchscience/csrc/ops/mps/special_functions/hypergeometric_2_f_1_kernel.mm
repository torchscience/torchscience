#include "../../special_functions.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science::ops::mps {

at::Tensor
hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(
        a.device().is_mps() || a.device().is_cpu(),
        "a must be an MPS or CPU tensor"
    );

    auto a_cpu = a.to(at::kCPU);
    auto b_cpu = b.to(at::kCPU);
    auto c_cpu = c.to(at::kCPU);
    auto z_cpu = z.to(at::kCPU);

    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "torchscience::hypergeometric_2_f_1",
        ""
    ).typed<at::Tensor(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&
    )>();

    auto result_cpu = op.call(a_cpu, b_cpu, c_cpu, z_cpu);

    return result_cpu.to(a.device());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hypergeometric_2_f_1_backward_kernel(
    const at::Tensor& grad_out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z,
    const at::Tensor& result
) {
    auto grad_out_cpu = grad_out.to(at::kCPU);

    auto a_cpu = a.to(at::kCPU);
    auto b_cpu = b.to(at::kCPU);
    auto c_cpu = c.to(at::kCPU);
    auto z_cpu = z.to(at::kCPU);

    auto result_cpu = result.to(at::kCPU);

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

    auto grads_cpu = op.call(
        grad_out_cpu,
        a_cpu,
        b_cpu,
        c_cpu,
        z_cpu,
        result_cpu
    );

    return std::make_tuple(
        std::get<0>(grads_cpu).to(a.device()),
        std::get<1>(grads_cpu).to(b.device()),
        std::get<2>(grads_cpu).to(c.device()),
        std::get<3>(grads_cpu).to(z.device())
    );
}

} // namespace science::ops::mps

TORCH_LIBRARY_IMPL(torchscience, MPS, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            science::ops::mps::hypergeometric_2_f_1_forward_kernel
        )
    );

    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::_hypergeometric_2_f_1_backward"
        ),
        TORCH_FN(
            science::ops::mps::hypergeometric_2_f_1_backward_kernel
        )
    );
}
