#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::special_functions {

inline at::ScalarType compute_output_dtype(
  const at::ScalarType v_dtype,
  const at::ScalarType z_dtype
) {
  const bool v_complex = isComplexType(v_dtype);
  bool z_complex = isComplexType(z_dtype);

  if (v_complex || z_complex) {
    c10::ScalarType v_real;

    if (v_complex) {
      v_real = toRealValueType(v_dtype);
    } else {
      v_real = v_dtype;
    }

    c10::ScalarType z_real;

    if (z_complex) {
      z_real = toRealValueType(z_dtype);
    } else {
      z_real = z_dtype;
    }

    if (at::promote_types(v_real, z_real) == at::kDouble) {
      return at::kComplexDouble;
    }

    return at::kComplexFloat;
  }

  return at::promote_types(v_dtype, z_dtype);
}

inline at::Tensor chebyshev_polynomial_t(const at::Tensor& v, const at::Tensor& z) {
  auto output_size = at::infer_size(v.sizes(), z.sizes());
  auto output_dtype = compute_output_dtype(v.scalar_type(), z.scalar_type());
  return at::empty(output_size, z.options().dtype(output_dtype).device(at::kMeta));
}

inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
    const at::Tensor& gradient_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  const auto output_size = at::infer_size(
    at::infer_size(gradient_output.sizes(), v.sizes()),
    z.sizes()
  );

  auto output_dtype = compute_output_dtype(gradient_output.scalar_type(), compute_output_dtype(v.scalar_type(), z.scalar_type()));

  auto gradient_v = at::empty(output_size, v.options().dtype(output_dtype).device(at::kMeta));
  auto gradient_z = at::empty(output_size, z.options().dtype(output_dtype).device(at::kMeta));

  return std::make_tuple(
    gradient_v,
    gradient_z
  );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
    const at::Tensor& gradient_gradient_v,
    const at::Tensor& gradient_gradient_z,
    const at::Tensor& gradient_output,
    const at::Tensor& v,
    const at::Tensor& z
) {
  const bool has_ggv = gradient_gradient_v.defined();
  const bool has_ggz = gradient_gradient_z.defined();

  auto output_size = at::infer_size(gradient_output.sizes(), at::infer_size(v.sizes(), z.sizes()));
  if (has_ggv) {
    output_size = at::infer_size(output_size, gradient_gradient_v.sizes());
  }
  if (has_ggz) {
    output_size = at::infer_size(output_size, gradient_gradient_z.sizes());
  }

  auto promoted_dtype = compute_output_dtype(
      compute_output_dtype(gradient_output.scalar_type(), v.scalar_type()),
      z.scalar_type());
  if (has_ggv) {
    promoted_dtype = compute_output_dtype(promoted_dtype, gradient_gradient_v.scalar_type());
  }
  if (has_ggz) {
    promoted_dtype = compute_output_dtype(promoted_dtype, gradient_gradient_z.scalar_type());
  }

  auto gradient_gradient_output = at::empty(output_size, gradient_output.options().dtype(promoted_dtype).device(at::kMeta));
  auto gradient_v = at::empty(output_size, v.options().dtype(promoted_dtype).device(at::kMeta));
  auto gradient_z = at::empty(output_size, z.options().dtype(promoted_dtype).device(at::kMeta));

  return std::make_tuple(
    gradient_gradient_output,
    gradient_v,
    gradient_z
  );
}

}  // namespace torchscience::meta::special_functions

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
  module.impl(
    "chebyshev_polynomial_t",
    &torchscience::meta::special_functions::chebyshev_polynomial_t
  );

  module.impl(
    "chebyshev_polynomial_t_backward",
    &torchscience::meta::special_functions::chebyshev_polynomial_t_backward
  );

  module.impl(
    "chebyshev_polynomial_t_backward_backward",
    &torchscience::meta::special_functions::chebyshev_polynomial_t_backward_backward
  );
}
