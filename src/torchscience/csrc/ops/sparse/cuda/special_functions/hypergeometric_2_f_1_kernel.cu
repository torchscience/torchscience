#include "../../../special_functions.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science {
namespace ops {
namespace sparse {
namespace cuda {

at::Tensor
hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(
        z.is_sparse(),
        "z must be a sparse tensor for SparseCUDA dispatch"
    );

    auto z_indices = z._indices();
    auto z_values = z._values();

    auto a_values = a.is_sparse() ? a._values() : a;
    auto b_values = b.is_sparse() ? b._values() : b;
    auto c_values = c.is_sparse() ? c._values() : c;

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

    // Reconstruct sparse tensor with same indices, new values
    return at::sparse_coo_tensor(z_indices, result_values, z.sizes(), z.options());
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
    auto grad_out_values = grad_out.is_sparse() ? grad_out._values() : grad_out;

    auto a_values = a.is_sparse() ? a._values() : a;
    auto b_values = b.is_sparse() ? b._values() : b;
    auto c_values = c.is_sparse() ? c._values() : c;
    auto z_values = z.is_sparse() ? z._values() : z;

    auto result_values = result.is_sparse() ? result._values() : result;

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

    auto grads_values = op.call(
        grad_out_values,
        a_values,
        b_values,
        c_values,
        z_values,
        result_values
    );

    // Reconstruct sparse gradients if inputs were sparse
    auto grad_a = a.is_sparse()
        ? at::sparse_coo_tensor(a._indices(), std::get<0>(grads_values), a.sizes(), a.options())
        : std::get<0>(grads_values);

    auto grad_b = b.is_sparse()
        ? at::sparse_coo_tensor(b._indices(), std::get<1>(grads_values), b.sizes(), b.options())
        : std::get<1>(grads_values);

    auto grad_c = c.is_sparse()
        ? at::sparse_coo_tensor(c._indices(), std::get<2>(grads_values), c.sizes(), c.options())
        : std::get<2>(grads_values);

    auto grad_z = z.is_sparse()
        ? at::sparse_coo_tensor(z._indices(), std::get<3>(grads_values), z.sizes(), z.options())
        : std::get<3>(grads_values);

    return std::make_tuple(grad_a, grad_b, grad_c, grad_z);
}

}  // namespace cuda
}  // namespace sparse
}  // namespace ops
}  // namespace science

TORCH_LIBRARY_IMPL(torchscience, SparseCUDA, module) {
    module.impl(TORCH_SELECTIVE_NAME(
        "torchscience::hypergeometric_2_f_1"),
        TORCH_FN(
            science::ops::sparse::cuda::hypergeometric_2_f_1_forward_kernel
        )
    );

    module.impl(
        TORCH_SELECTIVE_NAME("torchscience::_hypergeometric_2_f_1_backward"),
        TORCH_FN(
            science::ops::sparse::cuda::hypergeometric_2_f_1_backward_kernel
        )
    );
}
