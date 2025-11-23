#pragma once

#include <ATen/ATen.h>

namespace science::ops {

namespace cpu {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    hypergeometric_2_f_1_backward_kernel(
        const at::Tensor& grad_out,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z,
        const at::Tensor& result
    );
}  // namespace cpu

#ifdef WITH_CUDA
namespace cuda {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    hypergeometric_2_f_1_backward_kernel(
        const at::Tensor& grad_out,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z,
        const at::Tensor& result
    );
}  // namespace cuda
#endif

#ifdef __APPLE__
namespace mps {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    hypergeometric_2_f_1_backward_kernel(
        const at::Tensor& grad_out,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z,
        const at::Tensor& result
    );
}  // namespace mps
#endif

#ifdef WITH_HIP
namespace hip {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    hypergeometric_2_f_1_backward_kernel(
        const at::Tensor& grad_out,
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z,
        const at::Tensor& result
    );
}  // namespace hip
#endif

namespace sparse::cpu {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a, const at::Tensor& b,
        const at::Tensor& c, const at::Tensor& z);

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    hypergeometric_2_f_1_backward_kernel(
        const at::Tensor& grad_out,
        const at::Tensor& a, const at::Tensor& b,
        const at::Tensor& c, const at::Tensor& z,
        const at::Tensor& result);
} // namespace sparse::cpu

#ifdef WITH_CUDA
    namespace cuda {
        at::Tensor hypergeometric_2_f_1_forward_kernel(
            const at::Tensor& a,
            const at::Tensor& b,
            const at::Tensor& c,
            const at::Tensor& z
        );

        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
        hypergeometric_2_f_1_backward_kernel(
            const at::Tensor& grad_out,
            const at::Tensor& a,
            const at::Tensor& b,
            const at::Tensor& c,
            const at::Tensor& z,
            const at::Tensor& result
        );
    }  // namespace cuda
#endif

namespace quantized::cpu {
    at::Tensor hypergeometric_2_f_1_forward_kernel(
        const at::Tensor& a,
        const at::Tensor& b,
        const at::Tensor& c,
        const at::Tensor& z
    );
} // namespace quantized::cpu

at::Tensor hypergeometric_2_f_1_forward_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hypergeometric_2_f_1_backward_meta(
    const at::Tensor& grad_out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z,
    const at::Tensor& result
);

} // namespace science::ops
