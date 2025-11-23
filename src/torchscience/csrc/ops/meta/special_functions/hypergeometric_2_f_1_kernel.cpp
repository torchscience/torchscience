#include "../../special_functions.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace science::ops {

at::Tensor
hypergeometric_2_f_1_forward_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    const std::vector<at::Tensor> broadcasted = at::broadcast_tensors({a, b, c, z});

    const c10::ScalarType dtype_ab = promoteTypes(a.scalar_type(), b.scalar_type());
    const c10::ScalarType dtype_cz = promoteTypes(c.scalar_type(), z.scalar_type());

    c10::ScalarType promoted_dtype = promoteTypes(dtype_ab, dtype_cz);

    return at::empty(broadcasted[3].sizes(), at::TensorOptions().dtype(promoted_dtype).device(z.device()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hypergeometric_2_f_1_backward_meta(
    const at::Tensor& grad_out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z,
    const at::Tensor& result
) {
    at::Tensor grad_a = empty_like(a);
    at::Tensor grad_b = empty_like(b);
    at::Tensor grad_c = empty_like(c);
    at::Tensor grad_z = empty_like(z);

    return std::make_tuple(
        grad_a,
        grad_b,
        grad_c,
        grad_z
    );
}

} // namespace science::ops

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "hypergeometric_2_f_1",
        science::ops::hypergeometric_2_f_1_forward_meta
    );
    module.impl(
        "_hypergeometric_2_f_1_backward",
        science::ops::hypergeometric_2_f_1_backward_meta
    );
}
