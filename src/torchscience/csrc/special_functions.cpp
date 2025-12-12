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
  module.def("_bulirsch_elliptic_integral_el1(Tensor x, Tensor kc) -> Tensor");
  module.def("_bulirsch_elliptic_integral_el1_backward(Tensor grad_output, Tensor x, Tensor kc) -> (Tensor, Tensor)");
  module.def("_carlson_elliptic_r_c(Tensor x, Tensor y) -> Tensor");
  module.def("_carlson_elliptic_r_c_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  module.def("_complete_carlson_elliptic_r_f(Tensor x, Tensor y) -> Tensor");
  module.def("_complete_carlson_elliptic_r_f_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  module.def("_complete_carlson_elliptic_r_g(Tensor x, Tensor y) -> Tensor");
  module.def("_complete_carlson_elliptic_r_g_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  module.def("_complete_elliptic_integral_e(Tensor input) -> Tensor");
  module.def("_complete_elliptic_integral_e_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_complete_elliptic_integral_k(Tensor input) -> Tensor");
  module.def("_complete_elliptic_integral_k_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_complete_elliptic_integral_pi(Tensor n, Tensor k) -> Tensor");
  module.def("_complete_elliptic_integral_pi_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  module.def("_complete_legendre_elliptic_integral_d(Tensor input) -> Tensor");
  module.def("_complete_legendre_elliptic_integral_d_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_confluent_hypergeometric_0_f_1(Tensor b, Tensor z) -> Tensor");
  module.def("_confluent_hypergeometric_0_f_1_backward(Tensor grad_output, Tensor b, Tensor z) -> (Tensor, Tensor)");
  module.def("_cos_pi(Tensor input) -> Tensor");
  module.def("_cos_pi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_cosine_integral_ci(Tensor input) -> Tensor");
  module.def("_cosine_integral_ci_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_digamma(Tensor input) -> Tensor");
  module.def("_digamma_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_erf(Tensor input) -> Tensor");
  module.def("_erf_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_erfc(Tensor input) -> Tensor");
  module.def("_erfc_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_exponential_integral_e(Tensor n, Tensor x) -> Tensor");
  module.def("_exponential_integral_e_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_exponential_integral_ei(Tensor input) -> Tensor");
  module.def("_exponential_integral_ei_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_gamma(Tensor input) -> Tensor");
  module.def("_gamma_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_hankel_h_1(Tensor nu, Tensor x) -> Tensor");
  module.def("_hankel_h_1_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_hankel_h_2(Tensor nu, Tensor x) -> Tensor");
  module.def("_hankel_h_2_backward(Tensor grad_output, Tensor nu, Tensor x) -> (Tensor, Tensor)");
  module.def("_hyperbolic_cosine_integral_chi(Tensor input) -> Tensor");
  module.def("_hyperbolic_cosine_integral_chi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_hyperbolic_sine_integral_shi(Tensor input) -> Tensor");
  module.def("_hyperbolic_sine_integral_shi_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_incomplete_elliptic_integral_e(Tensor phi, Tensor k) -> Tensor");
  module.def("_incomplete_elliptic_integral_e_backward(Tensor grad_output, Tensor phi, Tensor k) -> (Tensor, Tensor)");
  module.def("_incomplete_elliptic_integral_f(Tensor phi, Tensor k) -> Tensor");
  module.def("_incomplete_elliptic_integral_f_backward(Tensor grad_output, Tensor phi, Tensor k) -> (Tensor, Tensor)");
  module.def("_incomplete_legendre_elliptic_integral_d(Tensor phi, Tensor k) -> Tensor");
  module.def("_incomplete_legendre_elliptic_integral_d_backward(Tensor grad_output, Tensor phi, Tensor k) -> (Tensor, Tensor)");
  module.def("_inverse_erf(Tensor input) -> Tensor");
  module.def("_inverse_erf_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_inverse_erfc(Tensor input) -> Tensor");
  module.def("_inverse_erfc_backward(Tensor grad_output, Tensor input) -> Tensor");
  module.def("_jacobi_theta_1(Tensor z, Tensor q) -> Tensor");
  module.def("_jacobi_theta_1_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
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
  module.def("_spherical_bessel_y(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_bessel_y_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_spherical_hankel_h_1(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_hankel_h_1_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_spherical_hankel_h_2(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_hankel_h_2_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_spherical_modified_bessel_i(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_modified_bessel_i_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_spherical_modified_bessel_k(Tensor n, Tensor x) -> Tensor");
  module.def("_spherical_modified_bessel_k_backward(Tensor grad_output, Tensor n, Tensor x) -> (Tensor, Tensor)");
  module.def("_trigamma(Tensor input) -> Tensor");
  module.def("_trigamma_backward(Tensor grad_output, Tensor input) -> Tensor");
}
