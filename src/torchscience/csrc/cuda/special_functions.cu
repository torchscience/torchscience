// CUDA kernels for high-use special functions
// This file is only compiled when CUDA is available (controlled by CMakeLists.txt)

// Gamma-related functions
#include <torchscience/csrc/cuda/special_functions/gamma.h>
#include <torchscience/csrc/cuda/special_functions/log_gamma.h>
#include <torchscience/csrc/cuda/special_functions/digamma.h>
#include <torchscience/csrc/cuda/special_functions/trigamma.h>
#include <torchscience/csrc/cuda/special_functions/beta.h>
#include <torchscience/csrc/cuda/special_functions/log_beta.h>

// Error functions
#include <torchscience/csrc/cuda/special_functions/error_erf.h>
#include <torchscience/csrc/cuda/special_functions/error_erfc.h>
#include <torchscience/csrc/cuda/special_functions/error_inverse_erf.h>
#include <torchscience/csrc/cuda/special_functions/error_inverse_erfc.h>

// Bessel functions
#include <torchscience/csrc/cuda/special_functions/bessel_j.h>
#include <torchscience/csrc/cuda/special_functions/bessel_j_derivative.h>
#include <torchscience/csrc/cuda/special_functions/bessel_y.h>
#include <torchscience/csrc/cuda/special_functions/bessel_y_derivative.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_i.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_i_derivative.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_k.h>
#include <torchscience/csrc/cuda/special_functions/modified_bessel_k_derivative.h>

// Airy functions
#include <torchscience/csrc/cuda/special_functions/airy_ai.h>
#include <torchscience/csrc/cuda/special_functions/airy_ai_derivative.h>
#include <torchscience/csrc/cuda/special_functions/airy_bi.h>
#include <torchscience/csrc/cuda/special_functions/airy_bi_derivative.h>

// Elliptic integrals and functions
#include <torchscience/csrc/cuda/special_functions/complete_elliptic_integral_k.h>
#include <torchscience/csrc/cuda/special_functions/complete_elliptic_integral_e.h>
#include <torchscience/csrc/cuda/special_functions/complete_legendre_elliptic_integral_d.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_sn.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_cn.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_dn.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_cd.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_sc.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_elliptic_sd.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_amplitude_am.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_sn.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_cn.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_dn.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_cd.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_sc.h>
#include <torchscience/csrc/cuda/special_functions/inverse_jacobi_elliptic_sd.h>

// Trigonometric functions
#include <torchscience/csrc/cuda/special_functions/sin_pi.h>
#include <torchscience/csrc/cuda/special_functions/cos_pi.h>
#include <torchscience/csrc/cuda/special_functions/sinc_pi.h>
#include <torchscience/csrc/cuda/special_functions/sinhc_pi.h>

// Exponential and logarithmic integrals
#include <torchscience/csrc/cuda/special_functions/exponential_integral_ei.h>
#include <torchscience/csrc/cuda/special_functions/exponential_integral_e_1.h>
#include <torchscience/csrc/cuda/special_functions/logarithmic_integral_li.h>

// Trigonometric integrals
#include <torchscience/csrc/cuda/special_functions/sine_integral_si.h>
#include <torchscience/csrc/cuda/special_functions/sine_integral_sin.h>
#include <torchscience/csrc/cuda/special_functions/cosine_integral_ci.h>
#include <torchscience/csrc/cuda/special_functions/cosine_integral_cin.h>
#include <torchscience/csrc/cuda/special_functions/hyperbolic_sine_integral_shi.h>
#include <torchscience/csrc/cuda/special_functions/hyperbolic_cosine_integral_chi.h>

// Factorial and combinatorial functions
#include <torchscience/csrc/cuda/special_functions/factorial.h>
#include <torchscience/csrc/cuda/special_functions/double_factorial.h>

// Number theory functions
#include <torchscience/csrc/cuda/special_functions/riemann_zeta.h>
#include <torchscience/csrc/cuda/special_functions/bernoulli_number_b.h>
#include <torchscience/csrc/cuda/special_functions/euler_number_e.h>
#include <torchscience/csrc/cuda/special_functions/euler_totient_phi.h>
#include <torchscience/csrc/cuda/special_functions/fibonacci_number_f.h>
#include <torchscience/csrc/cuda/special_functions/prime_number_p.h>
#include <torchscience/csrc/cuda/special_functions/mobius_mu.h>
#include <torchscience/csrc/cuda/special_functions/liouville_lambda.h>
#include <torchscience/csrc/cuda/special_functions/tangent_number_t_2.h>

// Polynomial functions
#include <torchscience/csrc/cuda/special_functions/polygamma.h>
#include <torchscience/csrc/cuda/special_functions/bernoulli_polynomial_b.h>
#include <torchscience/csrc/cuda/special_functions/euler_polynomial_e.h>
#include <torchscience/csrc/cuda/special_functions/chebyshev_polynomial_t.h>
#include <torchscience/csrc/cuda/special_functions/chebyshev_polynomial_u.h>
#include <torchscience/csrc/cuda/special_functions/chebyshev_polynomial_v.h>
#include <torchscience/csrc/cuda/special_functions/chebyshev_polynomial_w.h>
#include <torchscience/csrc/cuda/special_functions/shifted_chebyshev_polynomial_t.h>
#include <torchscience/csrc/cuda/special_functions/shifted_chebyshev_polynomial_u.h>
#include <torchscience/csrc/cuda/special_functions/shifted_chebyshev_polynomial_v.h>
#include <torchscience/csrc/cuda/special_functions/shifted_chebyshev_polynomial_w.h>
#include <torchscience/csrc/cuda/special_functions/hermite_polynomial_h.h>
#include <torchscience/csrc/cuda/special_functions/hermite_polynomial_he.h>
#include <torchscience/csrc/cuda/special_functions/legendre_p.h>
#include <torchscience/csrc/cuda/special_functions/legendre_q.h>
#include <torchscience/csrc/cuda/special_functions/associated_legendre_p.h>

// Additional Bessel and Hankel functions
#include <torchscience/csrc/cuda/special_functions/hankel_h_1.h>
#include <torchscience/csrc/cuda/special_functions/hankel_h_2.h>
#include <torchscience/csrc/cuda/special_functions/spherical_bessel_j.h>
#include <torchscience/csrc/cuda/special_functions/spherical_bessel_y.h>
#include <torchscience/csrc/cuda/special_functions/spherical_hankel_h_1.h>
#include <torchscience/csrc/cuda/special_functions/spherical_hankel_h_2.h>
#include <torchscience/csrc/cuda/special_functions/spherical_modified_bessel_i.h>
#include <torchscience/csrc/cuda/special_functions/spherical_modified_bessel_k.h>

// Kelvin functions
#include <torchscience/csrc/cuda/special_functions/kelvin_ber.h>
#include <torchscience/csrc/cuda/special_functions/kelvin_bei.h>
#include <torchscience/csrc/cuda/special_functions/kelvin_ker.h>
#include <torchscience/csrc/cuda/special_functions/kelvin_kei.h>

// Jacobi theta functions
#include <torchscience/csrc/cuda/special_functions/jacobi_theta_1.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_theta_2.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_theta_3.h>
#include <torchscience/csrc/cuda/special_functions/jacobi_theta_4.h>
#include <torchscience/csrc/cuda/special_functions/neville_theta_c.h>
#include <torchscience/csrc/cuda/special_functions/neville_theta_d.h>
#include <torchscience/csrc/cuda/special_functions/neville_theta_n.h>
#include <torchscience/csrc/cuda/special_functions/neville_theta_s.h>

// Carlson elliptic integrals
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_c.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_r_c.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_d.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_f.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_e.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_k.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_m.h>
#include <torchscience/csrc/cuda/special_functions/carlson_elliptic_integral_r_j.h>
#include <torchscience/csrc/cuda/special_functions/complete_carlson_elliptic_r_f.h>
#include <torchscience/csrc/cuda/special_functions/complete_carlson_elliptic_r_g.h>
#include <torchscience/csrc/cuda/special_functions/complete_elliptic_integral_pi.h>
#include <torchscience/csrc/cuda/special_functions/legendre_elliptic_integral_pi.h>

// Additional elliptic integrals
#include <torchscience/csrc/cuda/special_functions/bulirsch_elliptic_integral_el1.h>
#include <torchscience/csrc/cuda/special_functions/incomplete_elliptic_integral_e.h>
#include <torchscience/csrc/cuda/special_functions/incomplete_elliptic_integral_f.h>
#include <torchscience/csrc/cuda/special_functions/incomplete_legendre_elliptic_integral_d.h>

// Hypergeometric functions
#include <torchscience/csrc/cuda/special_functions/confluent_hypergeometric_0_f_1.h>
#include <torchscience/csrc/cuda/special_functions/confluent_hypergeometric_1_f_1.h>

// Special integrals
#include <torchscience/csrc/cuda/special_functions/exponential_integral_e.h>
#include <torchscience/csrc/cuda/special_functions/parabolic_cylinder_d.h>

// Whittaker functions
#include <torchscience/csrc/cuda/special_functions/whittaker_m.h>
#include <torchscience/csrc/cuda/special_functions/whittaker_w.h>

// Combinatorial functions
#include <torchscience/csrc/cuda/special_functions/binomial_coefficient.h>
#include <torchscience/csrc/cuda/special_functions/falling_factorial.h>
#include <torchscience/csrc/cuda/special_functions/rising_factorial.h>
#include <torchscience/csrc/cuda/special_functions/stirling_number_s_1.h>
#include <torchscience/csrc/cuda/special_functions/stirling_number_s_2.h>
