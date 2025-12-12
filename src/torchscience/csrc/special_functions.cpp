#include <torch/library.h>

#include <torchscience/csrc/autocast/special_functions.h>
#include <torchscience/csrc/autograd/special_functions.h>
#include <torchscience/csrc/cpu/special_functions.h>
#include <torchscience/csrc/meta/special_functions.h>
#include <torchscience/csrc/quantized/cpu/special_functions.h>
#include <torchscience/csrc/sparse/coo/cpu/special_functions.h>
#include <torchscience/csrc/sparse/csr/cpu/special_functions.h>

TORCH_LIBRARY_FRAGMENT(torchscience, module) {
  module.def("_airy_ai(Tensor input) -> Tensor");
  module.def("_airy_ai_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_airy_bi(Tensor input) -> Tensor");
  module.def("_airy_bi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_bessel_j(Tensor nu, Tensor x) -> Tensor");
  module.def("_bessel_j_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_bessel_y(Tensor nu, Tensor x) -> Tensor");
  module.def("_bessel_y_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_beta(Tensor a, Tensor b) -> Tensor");
  module.def("_beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("_cos_pi(Tensor input) -> Tensor");
  module.def("_cos_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_digamma(Tensor input) -> Tensor");
  module.def("_digamma_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_erf(Tensor input) -> Tensor");
  module.def("_erf_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_erfc(Tensor input) -> Tensor");
  module.def("_erfc_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_gamma(Tensor input) -> Tensor");
  module.def("_gamma_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_inverse_erf(Tensor input) -> Tensor");
  module.def("_inverse_erf_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_inverse_erfc(Tensor input) -> Tensor");
  module.def("_inverse_erfc_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_log_gamma(Tensor input) -> Tensor");
  module.def("_log_gamma_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_modified_bessel_i(Tensor nu, Tensor x) -> Tensor");
  module.def("_modified_bessel_i_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_modified_bessel_k(Tensor nu, Tensor x) -> Tensor");
  module.def("_modified_bessel_k_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_sin_pi(Tensor input) -> Tensor");
  module.def("_sin_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_sinc_pi(Tensor input) -> Tensor");
  module.def("_sinc_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_sinhc_pi(Tensor input) -> Tensor");
  module.def("_sinhc_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_spherical_bessel_j(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_bessel_j_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_trigamma(Tensor input) -> Tensor");
  module.def("_trigamma_backward(Tensor grad_output, Tensor input) -> Tensor");
}
