#include "../../../special_functions.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science::ops::sparse::cpu {

at::Tensor
hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(
        z.is_sparse(),
        "z must be a sparse tensor for SparseCPU dispatch"
    );

    // Extract sparse components
    auto z_indices = z._indices();
    auto z_values = z._values();

    // Extract values from other tensors (handle both dense and sparse)
    auto a_values = a.is_sparse() ? a._values() : a;
    auto b_values = b.is_sparse() ? b._values() : b;
    auto c_values = c.is_sparse() ? c._values() : c;

    // Call CPU implementation on values via dispatcher
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "torchscience::hypergeometric_2_f_1",
        ""
    ).typed<at::Tensor(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&
    )>();

    auto result_values = op.call(a_values, b_values, c_values, z_values);

    return at::sparse_coo_tensor(
        z_indices,
        result_values,
        z.sizes(),
        z.options()
    );
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
    const at::Tensor grad_out_values = grad_out.is_sparse() ? grad_out._values() : grad_out;

    const at::Tensor a_values = a.is_sparse() ? a._values() : a;
    const at::Tensor b_values = b.is_sparse() ? b._values() : b;
    const at::Tensor c_values = c.is_sparse() ? c._values() : c;
    const at::Tensor z_values = z.is_sparse() ? z._values() : z;

    const at::Tensor result_values = result.is_sparse() ? result._values() : result;

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

    const auto grads_values = op.call(
        grad_out_values,
        a_values,
        b_values,
        c_values,
        z_values,
        result_values
    );

    auto grad_a = a.is_sparse() ? at::sparse_coo_tensor(a._indices(), std::get<0>(grads_values), a.sizes(), a.options()) : std::get<0>(grads_values);

    auto grad_b = b.is_sparse() ? at::sparse_coo_tensor(b._indices(), std::get<1>(grads_values), b.sizes(), b.options()) : std::get<1>(grads_values);

    auto grad_c = c.is_sparse() ? at::sparse_coo_tensor(c._indices(), std::get<2>(grads_values), c.sizes(), c.options()) : std::get<2>(grads_values);

    auto grad_z = z.is_sparse() ? at::sparse_coo_tensor(z._indices(), std::get<3>(grads_values), z.sizes(), z.options()) : std::get<3>(grads_values);

    return std::make_tuple(
        grad_a,
        grad_b,
        grad_c,
        grad_z
    );
}

} // namespace science::ops::sparse::cpu




TORCH_LIBRARY_IMPL(torchscience, SparseCPU, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            science::ops::sparse::cpu::hypergeometric_2_f_1_forward_kernel
        )
    );

    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::_hypergeometric_2_f_1_backward"
        ),
        TORCH_FN(
            science::ops::sparse::cpu::hypergeometric_2_f_1_backward_kernel
        )
    );
}
