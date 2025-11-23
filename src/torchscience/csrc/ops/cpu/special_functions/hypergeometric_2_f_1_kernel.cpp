#include "../../special_functions.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

#include <cmath>
#include <complex>

namespace science {
namespace ops {
namespace cpu {

// Helper to get real type from scalar type
template <typename T>
struct real_type { using type = T; };

template <typename T>
struct real_type<c10::complex<T>> { using type = T; };

template <typename T>
using real_type_t = typename real_type<T>::type;

// Helper to get real part (works for both real and complex)
template <typename T> real_type_t<T> get_real(const T& val) {
    if constexpr (c10::is_complex<T>::value) {
        return val.real();
    } else {
        return val;
    }
}

template <typename scalar_t>
scalar_t
hypergeometric_2_f_1_impl(
    scalar_t a,
    scalar_t b,
    scalar_t c,
    scalar_t z,
    const int depth = 0
) {
    using std::abs;
    using std::log;
    using std::exp;
    using real_t = real_type_t<scalar_t>;

    if (constexpr int max_recursion_depth = 10; depth > max_recursion_depth) {
        goto series_expansion;
    }

    if (abs(z) < real_t(1e-14)) {
        return scalar_t(1.0);
    }

    if (abs(z) >= real_t(1.0)) {
        if (get_real(z) > real_t(0.5)) {
            if (scalar_t z_trans = z / (z - scalar_t(1.0)); abs(z_trans) < abs(z) && abs(z_trans) < real_t(1.0)) {
                scalar_t factor = std::pow(scalar_t(1.0) - z, -a);

                return factor * hypergeometric_2_f_1_impl(a, c - b, c, z_trans, depth + 1);
            }
        }
    }
series_expansion:
    const auto tolerance = real_t(1e-12);

    auto sum = scalar_t(1.0);
    auto term = scalar_t(1.0);

    for (int n = 0; n < 10000; n++) {
        scalar_t a_n = a + scalar_t(n);
        scalar_t b_n = b + scalar_t(n);
        scalar_t c_n = c + scalar_t(n);

        auto n_plus_1 = scalar_t(n + 1);

        term = term * (a_n * b_n * z) / (c_n * n_plus_1);

        sum = sum + term;

        if (abs(term) < tolerance * abs(sum)) {
            break;
        }
    }

    return sum;
}

at::Tensor
hypergeometric_2_f_1_forward_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& z
) {
    TORCH_CHECK(a.device().is_cpu(), "a must be a CPU tensor");
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor");
    TORCH_CHECK(c.device().is_cpu(), "c must be a CPU tensor");
    TORCH_CHECK(z.device().is_cpu(), "z must be a CPU tensor");

    const std::vector<at::Tensor> a_b = at::broadcast_tensors({a, b, c, z});

    const c10::ScalarType dtype_ab = promoteTypes(a.scalar_type(), b.scalar_type());
    const c10::ScalarType dtype_cz = promoteTypes(c.scalar_type(), z.scalar_type());
    const c10::ScalarType promoted_dtype = promoteTypes(dtype_ab, dtype_cz);

    const at::Tensor a_broad = a_b[0].to(promoted_dtype).contiguous();
    const at::Tensor b_broad = a_b[1].to(promoted_dtype).contiguous();
    const at::Tensor c_broad = a_b[2].to(promoted_dtype).contiguous();
    const at::Tensor z_broad = a_b[3].to(promoted_dtype).contiguous();

    at::Tensor result = empty_like(z_broad);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        result.scalar_type(),
        "hypergeometric_2_f_1_cpu",
        [&] {
            auto a_data = a_broad.data_ptr<scalar_t>();
            auto b_data = b_broad.data_ptr<scalar_t>();
            auto c_data = c_broad.data_ptr<scalar_t>();
            auto z_data = z_broad.data_ptr<scalar_t>();
            auto result_data = result.data_ptr<scalar_t>();

            int64_t numel = result.numel();

            for (int64_t i = 0; i < numel; i++) {
                result_data[i] = hypergeometric_2_f_1_impl(
                    a_data[i],
                    b_data[i],
                    c_data[i],
                    z_data[i]
                );
            }
        }
    );

    return result;
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
    at::Tensor grad_z = grad_out * (a * b / c) * hypergeometric_2_f_1_forward_kernel(a + 1.0, b + 1.0, c + 1.0, z);
    at::Tensor grad_a = grad_out * (hypergeometric_2_f_1_forward_kernel(a + 1e-4, b, c, z) - result) / 1e-4;
    at::Tensor grad_b = grad_out * (hypergeometric_2_f_1_forward_kernel(a, b + 1e-4, c, z) - result) / 1e-4;
    at::Tensor grad_c = grad_out * (hypergeometric_2_f_1_forward_kernel(a, b, c + 1e-4, z) - result) / 1e-4;

    return std::make_tuple(
        grad_a,
        grad_b,
        grad_c,
        grad_z
    );
}

}  // namespace cpu
}  // namespace ops
}  // namespace science

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::hypergeometric_2_f_1"
        ),
        TORCH_FN(
            science::ops::cpu::hypergeometric_2_f_1_forward_kernel
        )
    );

    module.impl(
        TORCH_SELECTIVE_NAME(
            "torchscience::_hypergeometric_2_f_1_backward"
        ),
        TORCH_FN(
            science::ops::cpu::hypergeometric_2_f_1_backward_kernel
        )
    );
}
