// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include "macros.cuh"

// Gamma function and derivatives
#include "../kernel/special_functions/gamma.h"
#include "../kernel/special_functions/gamma_backward.h"
#include "../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(gamma, z)

#include "../kernel/special_functions/digamma.h"
#include "../kernel/special_functions/digamma_backward.h"
#include "../kernel/special_functions/digamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(digamma, z)

#include "../kernel/special_functions/trigamma.h"
#include "../kernel/special_functions/trigamma_backward.h"
#include "../kernel/special_functions/trigamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(trigamma, z)

#include "../kernel/special_functions/log_gamma.h"
#include "../kernel/special_functions/log_gamma_backward.h"
#include "../kernel/special_functions/log_gamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(log_gamma, z)

#include "../kernel/special_functions/reciprocal_gamma.h"
#include "../kernel/special_functions/reciprocal_gamma_backward.h"
#include "../kernel/special_functions/reciprocal_gamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(reciprocal_gamma, z)

#include "../kernel/special_functions/gamma_sign.h"
#include "../kernel/special_functions/gamma_sign_backward.h"
#include "../kernel/special_functions/gamma_sign_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(gamma_sign, x)

// Beta function
#include "../kernel/special_functions/beta.h"
#include "../kernel/special_functions/beta_backward.h"
#include "../kernel/special_functions/beta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(beta, a, b)

#include "../kernel/special_functions/log_beta.h"
#include "../kernel/special_functions/log_beta_backward.h"
#include "../kernel/special_functions/log_beta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(log_beta, a, b)

#include "../kernel/special_functions/incomplete_beta.h"
#include "../kernel/special_functions/incomplete_beta_backward.h"
#include "../kernel/special_functions/incomplete_beta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(incomplete_beta, x, a, b)

// Polygamma
#include "../kernel/special_functions/polygamma.h"
#include "../kernel/special_functions/polygamma_backward.h"
#include "../kernel/special_functions/polygamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(polygamma, n, z)

// Regularized gamma functions
#include "../kernel/special_functions/regularized_gamma_p.h"
#include "../kernel/special_functions/regularized_gamma_p_backward.h"
#include "../kernel/special_functions/regularized_gamma_p_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(regularized_gamma_p, a, x)

#include "../kernel/special_functions/regularized_gamma_q.h"
#include "../kernel/special_functions/regularized_gamma_q_backward.h"
#include "../kernel/special_functions/regularized_gamma_q_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(regularized_gamma_q, a, x)

// Inverse regularized gamma/beta functions
#include "../kernel/special_functions/inverse_regularized_gamma_p.h"
#include "../kernel/special_functions/inverse_regularized_gamma_p_backward.h"
#include "../kernel/special_functions/inverse_regularized_gamma_p_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_p, a, y)

#include "../kernel/special_functions/inverse_regularized_gamma_q.h"
#include "../kernel/special_functions/inverse_regularized_gamma_q_backward.h"
#include "../kernel/special_functions/inverse_regularized_gamma_q_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_q, a, y)

#include "../kernel/special_functions/inverse_regularized_incomplete_beta.h"
#include "../kernel/special_functions/inverse_regularized_incomplete_beta_backward.h"
#include "../kernel/special_functions/inverse_regularized_incomplete_beta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(inverse_regularized_incomplete_beta, a, b, y)

#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta.h"
#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta_backward.h"
#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(inverse_complementary_regularized_incomplete_beta, a, b, y)

// Hypergeometric functions
#include "../kernel/special_functions/hypergeometric_2_f_1.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_2_f_1, a, b, c, z)

#include "../kernel/special_functions/confluent_hypergeometric_m.h"
#include "../kernel/special_functions/confluent_hypergeometric_m_backward.h"
#include "../kernel/special_functions/confluent_hypergeometric_m_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_m, a, b, z)

#include "../kernel/special_functions/confluent_hypergeometric_u.h"
#include "../kernel/special_functions/confluent_hypergeometric_u_backward.h"
#include "../kernel/special_functions/confluent_hypergeometric_u_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_u, a, b, z)

#include "../kernel/special_functions/hypergeometric_0_f_1.h"
#include "../kernel/special_functions/hypergeometric_0_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_0_f_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(hypergeometric_0_f_1, b, z)

#include "../kernel/special_functions/hypergeometric_1_f_2.h"
#include "../kernel/special_functions/hypergeometric_1_f_2_backward.h"
#include "../kernel/special_functions/hypergeometric_1_f_2_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_1_f_2, a, b1, b2, z)

// Whittaker functions
#include "../kernel/special_functions/whittaker_m.h"
#include "../kernel/special_functions/whittaker_m_backward.h"
#include "../kernel/special_functions/whittaker_m_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(whittaker_m, kappa, mu, z)

#include "../kernel/special_functions/whittaker_w.h"
#include "../kernel/special_functions/whittaker_w_backward.h"
#include "../kernel/special_functions/whittaker_w_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(whittaker_w, kappa, mu, z)

// Chebyshev polynomials
#include "../kernel/special_functions/chebyshev_polynomial_t.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)

#include "../kernel/special_functions/chebyshev_polynomial_u.h"
#include "../kernel/special_functions/chebyshev_polynomial_u_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_u_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_u, x, n)

// Bessel functions of the first kind
#include "../kernel/special_functions/bessel_j_0.h"
#include "../kernel/special_functions/bessel_j_0_backward.h"
#include "../kernel/special_functions/bessel_j_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(bessel_j_0, z)

#include "../kernel/special_functions/bessel_j_1.h"
#include "../kernel/special_functions/bessel_j_1_backward.h"
#include "../kernel/special_functions/bessel_j_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(bessel_j_1, z)

#include "../kernel/special_functions/bessel_j.h"
#include "../kernel/special_functions/bessel_j_backward.h"
#include "../kernel/special_functions/bessel_j_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(bessel_j, n, z)

// Bessel functions of the second kind
#include "../kernel/special_functions/bessel_y_0.h"
#include "../kernel/special_functions/bessel_y_0_backward.h"
#include "../kernel/special_functions/bessel_y_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(bessel_y_0, z)

#include "../kernel/special_functions/bessel_y_1.h"
#include "../kernel/special_functions/bessel_y_1_backward.h"
#include "../kernel/special_functions/bessel_y_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(bessel_y_1, z)

#include "../kernel/special_functions/bessel_y.h"
#include "../kernel/special_functions/bessel_y_backward.h"
#include "../kernel/special_functions/bessel_y_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(bessel_y, n, z)

// Modified Bessel functions of the first kind
#include "../kernel/special_functions/modified_bessel_i_0.h"
#include "../kernel/special_functions/modified_bessel_i_0_backward.h"
#include "../kernel/special_functions/modified_bessel_i_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(modified_bessel_i_0, z)

#include "../kernel/special_functions/modified_bessel_i_1.h"
#include "../kernel/special_functions/modified_bessel_i_1_backward.h"
#include "../kernel/special_functions/modified_bessel_i_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(modified_bessel_i_1, z)

#include "../kernel/special_functions/modified_bessel_i.h"
#include "../kernel/special_functions/modified_bessel_i_backward.h"
#include "../kernel/special_functions/modified_bessel_i_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(modified_bessel_i, n, z)

// Modified Bessel functions of the second kind
#include "../kernel/special_functions/modified_bessel_k_0.h"
#include "../kernel/special_functions/modified_bessel_k_0_backward.h"
#include "../kernel/special_functions/modified_bessel_k_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(modified_bessel_k_0, z)

#include "../kernel/special_functions/modified_bessel_k_1.h"
#include "../kernel/special_functions/modified_bessel_k_1_backward.h"
#include "../kernel/special_functions/modified_bessel_k_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(modified_bessel_k_1, z)

#include "../kernel/special_functions/modified_bessel_k.h"
#include "../kernel/special_functions/modified_bessel_k_backward.h"
#include "../kernel/special_functions/modified_bessel_k_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(modified_bessel_k, n, z)

// Spherical Bessel functions of the first kind
#include "../kernel/special_functions/spherical_bessel_j_0.h"
#include "../kernel/special_functions/spherical_bessel_j_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_0, z)

#include "../kernel/special_functions/spherical_bessel_j_1.h"
#include "../kernel/special_functions/spherical_bessel_j_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_1, z)

#include "../kernel/special_functions/spherical_bessel_j.h"
#include "../kernel/special_functions/spherical_bessel_j_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(spherical_bessel_j, n, z)

// Spherical Bessel functions of the second kind
#include "../kernel/special_functions/spherical_bessel_y_0.h"
#include "../kernel/special_functions/spherical_bessel_y_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_0, z)

#include "../kernel/special_functions/spherical_bessel_y_1.h"
#include "../kernel/special_functions/spherical_bessel_y_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_1, z)

#include "../kernel/special_functions/spherical_bessel_y.h"
#include "../kernel/special_functions/spherical_bessel_y_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(spherical_bessel_y, n, z)

// Modified spherical Bessel functions of the first kind
#include "../kernel/special_functions/spherical_bessel_i_0.h"
#include "../kernel/special_functions/spherical_bessel_i_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_0, z)

#include "../kernel/special_functions/spherical_bessel_i_1.h"
#include "../kernel/special_functions/spherical_bessel_i_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_1, z)

#include "../kernel/special_functions/spherical_bessel_i.h"
#include "../kernel/special_functions/spherical_bessel_i_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(spherical_bessel_i, n, z)

// Modified spherical Bessel functions of the second kind
#include "../kernel/special_functions/spherical_bessel_k_0.h"
#include "../kernel/special_functions/spherical_bessel_k_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_0, z)

#include "../kernel/special_functions/spherical_bessel_k_1.h"
#include "../kernel/special_functions/spherical_bessel_k_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_1, z)

#include "../kernel/special_functions/spherical_bessel_k.h"
#include "../kernel/special_functions/spherical_bessel_k_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(spherical_bessel_k, n, z)

// Carlson elliptic integrals
#include "../kernel/special_functions/carlson_elliptic_integral_r_f.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_f_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_f_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_f, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_d.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_d_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_d_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_d, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_c.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_c_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_c_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_c, x, y)

#include "../kernel/special_functions/carlson_elliptic_integral_r_j.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_j_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_j_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(carlson_elliptic_integral_r_j, x, y, z, p)

#include "../kernel/special_functions/carlson_elliptic_integral_r_g.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_g_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_g_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_g, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_e.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_e_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_e_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_e, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_m.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_m_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_m_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_m, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_k.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_k_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_k_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_k, x, y)

// Legendre elliptic integrals
#include "../kernel/special_functions/complete_legendre_elliptic_integral_k.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_k_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_k_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_k, m)

#include "../kernel/special_functions/complete_legendre_elliptic_integral_e.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_e_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_e_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_e, m)

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_e, phi, m)

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_f, phi, m)

#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(complete_legendre_elliptic_integral_pi, n, m)

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(incomplete_legendre_elliptic_integral_pi, n, phi, m)

// Jacobi elliptic functions
#include "../kernel/special_functions/jacobi_amplitude_am.h"
#include "../kernel/special_functions/jacobi_amplitude_am_backward.h"
#include "../kernel/special_functions/jacobi_amplitude_am_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_amplitude_am, u, m)

#include "../kernel/special_functions/jacobi_elliptic_dn.h"
#include "../kernel/special_functions/jacobi_elliptic_dn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_dn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cn.h"
#include "../kernel/special_functions/jacobi_elliptic_cn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sn.h"
#include "../kernel/special_functions/jacobi_elliptic_sn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sd.h"
#include "../kernel/special_functions/jacobi_elliptic_sd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sd_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cd.h"
#include "../kernel/special_functions/jacobi_elliptic_cd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cd_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sc.h"
#include "../kernel/special_functions/jacobi_elliptic_sc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sc_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_nd.h"
#include "../kernel/special_functions/jacobi_elliptic_nd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_nd_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_nc.h"
#include "../kernel/special_functions/jacobi_elliptic_nc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_nc_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_ns.h"
#include "../kernel/special_functions/jacobi_elliptic_ns_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_ns_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ns, u, m)

#include "../kernel/special_functions/jacobi_elliptic_dc.h"
#include "../kernel/special_functions/jacobi_elliptic_dc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_dc_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_ds.h"
#include "../kernel/special_functions/jacobi_elliptic_ds_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_ds_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ds, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cs.h"
#include "../kernel/special_functions/jacobi_elliptic_cs_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cs_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cs, u, m)

// Inverse Jacobi elliptic functions
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sn, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_cn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cn, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_dn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_dn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_dn_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_dn, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_sd.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sd_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sd_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sd, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_cd.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cd_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cd_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cd, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_sc.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sc_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sc_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sc, x, m)

// Jacobi theta functions
#include "../kernel/special_functions/theta_1.h"
#include "../kernel/special_functions/theta_1_backward.h"
#include "../kernel/special_functions/theta_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(theta_1, z, q)

#include "../kernel/special_functions/theta_2.h"
#include "../kernel/special_functions/theta_2_backward.h"
#include "../kernel/special_functions/theta_2_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(theta_2, z, q)

#include "../kernel/special_functions/theta_3.h"
#include "../kernel/special_functions/theta_3_backward.h"
#include "../kernel/special_functions/theta_3_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(theta_3, z, q)

#include "../kernel/special_functions/theta_4.h"
#include "../kernel/special_functions/theta_4_backward.h"
#include "../kernel/special_functions/theta_4_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(theta_4, z, q)

// Exponential integrals
#include "../kernel/special_functions/exponential_integral_ei.h"
#include "../kernel/special_functions/exponential_integral_ei_backward.h"
#include "../kernel/special_functions/exponential_integral_ei_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(exponential_integral_ei, x)

#include "../kernel/special_functions/exponential_integral_e_1.h"
#include "../kernel/special_functions/exponential_integral_e_1_backward.h"
#include "../kernel/special_functions/exponential_integral_e_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(exponential_integral_e_1, x)

#include "../kernel/special_functions/exponential_integral_ein.h"
#include "../kernel/special_functions/exponential_integral_ein_backward.h"
#include "../kernel/special_functions/exponential_integral_ein_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(exponential_integral_ein, x)

#include "../kernel/special_functions/exponential_integral_e.h"
#include "../kernel/special_functions/exponential_integral_e_backward.h"
#include "../kernel/special_functions/exponential_integral_e_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(exponential_integral_e, n, x)

// Sine and cosine integrals
#include "../kernel/special_functions/sine_integral_si.h"
#include "../kernel/special_functions/sine_integral_si_backward.h"
#include "../kernel/special_functions/sine_integral_si_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(sine_integral_si, x)

#include "../kernel/special_functions/cosine_integral_ci.h"
#include "../kernel/special_functions/cosine_integral_ci_backward.h"
#include "../kernel/special_functions/cosine_integral_ci_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(cosine_integral_ci, x)

// Airy functions
#include "../kernel/special_functions/airy_ai.h"
#include "../kernel/special_functions/airy_ai_backward.h"
#include "../kernel/special_functions/airy_ai_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(airy_ai, x)

#include "../kernel/special_functions/airy_bi.h"
#include "../kernel/special_functions/airy_bi_backward.h"
#include "../kernel/special_functions/airy_bi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(airy_bi, x)

// Lambert W function
#include "../kernel/special_functions/lambert_w.h"
#include "../kernel/special_functions/lambert_w_backward.h"
#include "../kernel/special_functions/lambert_w_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(lambert_w, k, z)

// Kelvin functions
#include "../kernel/special_functions/kelvin_ber.h"
#include "../kernel/special_functions/kelvin_ber_backward.h"
#include "../kernel/special_functions/kelvin_ber_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(kelvin_ber, x)

#include "../kernel/special_functions/kelvin_bei.h"
#include "../kernel/special_functions/kelvin_bei_backward.h"
#include "../kernel/special_functions/kelvin_bei_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(kelvin_bei, x)

#include "../kernel/special_functions/kelvin_ker.h"
#include "../kernel/special_functions/kelvin_ker_backward.h"
#include "../kernel/special_functions/kelvin_ker_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(kelvin_ker, x)

#include "../kernel/special_functions/kelvin_kei.h"
#include "../kernel/special_functions/kelvin_kei_backward.h"
#include "../kernel/special_functions/kelvin_kei_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(kelvin_kei, x)

// Riemann zeta function
#include "../kernel/special_functions/zeta.h"
#include "../kernel/special_functions/zeta_backward.h"
#include "../kernel/special_functions/zeta_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(zeta, s)

// Polylogarithm
#include "../kernel/special_functions/polylogarithm_li.h"
#include "../kernel/special_functions/polylogarithm_li_backward.h"
#include "../kernel/special_functions/polylogarithm_li_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(polylogarithm_li, s, z)

// Parabolic cylinder functions
#include "../kernel/special_functions/parabolic_cylinder_u.h"
#include "../kernel/special_functions/parabolic_cylinder_u_backward.h"
#include "../kernel/special_functions/parabolic_cylinder_u_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_u, a, z)

#include "../kernel/special_functions/parabolic_cylinder_v.h"
#include "../kernel/special_functions/parabolic_cylinder_v_backward.h"
#include "../kernel/special_functions/parabolic_cylinder_v_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_v, a, z)

// Error functions
#include "../kernel/special_functions/erfinv.h"
#include "../kernel/special_functions/erfinv_backward.h"
#include "../kernel/special_functions/erfinv_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(erfinv, x)

#include "../kernel/special_functions/erfcinv.h"
#include "../kernel/special_functions/erfcinv_backward.h"
#include "../kernel/special_functions/erfcinv_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(erfcinv, x)

// Fresnel integrals
#include "../kernel/special_functions/fresnel_s.h"
#include "../kernel/special_functions/fresnel_s_backward.h"
#include "../kernel/special_functions/fresnel_s_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(fresnel_s, z)

#include "../kernel/special_functions/fresnel_c.h"
#include "../kernel/special_functions/fresnel_c_backward.h"
#include "../kernel/special_functions/fresnel_c_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(fresnel_c, z)

// Orthogonal polynomials
#include "../kernel/special_functions/legendre_polynomial_p.h"
#include "../kernel/special_functions/legendre_polynomial_p_backward.h"
#include "../kernel/special_functions/legendre_polynomial_p_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(legendre_polynomial_p, n, z)

#include "../kernel/special_functions/legendre_polynomial_q.h"
#include "../kernel/special_functions/legendre_polynomial_q_backward.h"
#include "../kernel/special_functions/legendre_polynomial_q_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(legendre_polynomial_q, x, n)

// Associated Legendre polynomial P_n^m(x)
#include "../kernel/special_functions/associated_legendre_polynomial_p.h"
#include "../kernel/special_functions/associated_legendre_polynomial_p_backward.h"
#include "../kernel/special_functions/associated_legendre_polynomial_p_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(associated_legendre_polynomial_p, n, m, x)

#include "../kernel/special_functions/hermite_polynomial_h.h"
#include "../kernel/special_functions/hermite_polynomial_h_backward.h"
#include "../kernel/special_functions/hermite_polynomial_h_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(hermite_polynomial_h, n, z)

#include "../kernel/special_functions/hermite_polynomial_he.h"
#include "../kernel/special_functions/hermite_polynomial_he_backward.h"
#include "../kernel/special_functions/hermite_polynomial_he_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(hermite_polynomial_he, n, z)

#include "../kernel/special_functions/laguerre_polynomial_l.h"
#include "../kernel/special_functions/laguerre_polynomial_l_backward.h"
#include "../kernel/special_functions/laguerre_polynomial_l_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(laguerre_polynomial_l, n, alpha, z)

#include "../kernel/special_functions/gegenbauer_polynomial_c.h"
#include "../kernel/special_functions/gegenbauer_polynomial_c_backward.h"
#include "../kernel/special_functions/gegenbauer_polynomial_c_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(gegenbauer_polynomial_c, n, lambda, z)

#include "../kernel/special_functions/jacobi_polynomial_p.h"
#include "../kernel/special_functions/jacobi_polynomial_p_backward.h"
#include "../kernel/special_functions/jacobi_polynomial_p_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(jacobi_polynomial_p, n, alpha, beta, z)

#include "../kernel/special_functions/zernike_polynomial_r.h"
#include "../kernel/special_functions/zernike_polynomial_r_backward.h"
#include "../kernel/special_functions/zernike_polynomial_r_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(zernike_polynomial_r, n, m, rho)

#include "../kernel/special_functions/zernike_polynomial_z.h"
#include "../kernel/special_functions/zernike_polynomial_z_backward.h"
#include "../kernel/special_functions/zernike_polynomial_z_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(zernike_polynomial_z, n, m, rho, theta)

#include "../kernel/special_functions/krawtchouk_polynomial_k.h"
#include "../kernel/special_functions/krawtchouk_polynomial_k_backward.h"
#include "../kernel/special_functions/krawtchouk_polynomial_k_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(krawtchouk_polynomial_k, n, x, p, N)

#include "../kernel/special_functions/meixner_polynomial_m.h"
#include "../kernel/special_functions/meixner_polynomial_m_backward.h"
#include "../kernel/special_functions/meixner_polynomial_m_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUATERNARY_OPERATOR(meixner_polynomial_m, n, x, beta, c)

#include "../kernel/special_functions/charlier_polynomial_c.h"
#include "../kernel/special_functions/charlier_polynomial_c_backward.h"
#include "../kernel/special_functions/charlier_polynomial_c_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_TERNARY_OPERATOR(charlier_polynomial_c, n, x, a)

// Pochhammer symbol
#include "../kernel/special_functions/pochhammer.h"
#include "../kernel/special_functions/pochhammer_backward.h"
#include "../kernel/special_functions/pochhammer_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(pochhammer, z, m)

// Struve functions
#include "../kernel/special_functions/struve_h_0.h"
#include "../kernel/special_functions/struve_h_0_backward.h"
#include "../kernel/special_functions/struve_h_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(struve_h_0, z)

#include "../kernel/special_functions/struve_h_1.h"
#include "../kernel/special_functions/struve_h_1_backward.h"
#include "../kernel/special_functions/struve_h_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(struve_h_1, z)

#include "../kernel/special_functions/struve_l_0.h"
#include "../kernel/special_functions/struve_l_0_backward.h"
#include "../kernel/special_functions/struve_l_0_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(struve_l_0, z)

#include "../kernel/special_functions/struve_l_1.h"
#include "../kernel/special_functions/struve_l_1_backward.h"
#include "../kernel/special_functions/struve_l_1_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(struve_l_1, z)

#include "../kernel/special_functions/struve_h.h"
#include "../kernel/special_functions/struve_h_backward.h"
#include "../kernel/special_functions/struve_h_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(struve_h, n, z)

#include "../kernel/special_functions/struve_l.h"
#include "../kernel/special_functions/struve_l_backward.h"
#include "../kernel/special_functions/struve_l_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(struve_l, n, z)

// Anger and Weber functions
#include "../kernel/special_functions/anger_j.h"
#include "../kernel/special_functions/anger_j_backward.h"
#include "../kernel/special_functions/anger_j_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(anger_j, n, z)

#include "../kernel/special_functions/weber_e.h"
#include "../kernel/special_functions/weber_e_backward.h"
#include "../kernel/special_functions/weber_e_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(weber_e, n, z)

// Chebyshev polynomial of the third kind V_n(x)
#include "../kernel/special_functions/chebyshev_polynomial_v.h"
#include "../kernel/special_functions/chebyshev_polynomial_v_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_v_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_v, x, n)

// Chebyshev polynomial of the fourth kind W_n(x)
#include "../kernel/special_functions/chebyshev_polynomial_w.h"
#include "../kernel/special_functions/chebyshev_polynomial_w_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_w_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_w, x, n)

// Tetragamma function psi''(z)
#include "../kernel/special_functions/tetragamma.h"
#include "../kernel/special_functions/tetragamma_backward.h"
#include "../kernel/special_functions/tetragamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(tetragamma, z)

// Pentagamma function psi'''(z)
#include "../kernel/special_functions/pentagamma.h"
#include "../kernel/special_functions/pentagamma_backward.h"
#include "../kernel/special_functions/pentagamma_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(pentagamma, z)

// cos(pi * x)
#include "../kernel/special_functions/cos_pi.h"
#include "../kernel/special_functions/cos_pi_backward.h"
#include "../kernel/special_functions/cos_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(cos_pi, x)

// cosh(pi * x)
#include "../kernel/special_functions/cosh_pi.h"
#include "../kernel/special_functions/cosh_pi_backward.h"
#include "../kernel/special_functions/cosh_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(cosh_pi, x)

// sin(pi * x)
#include "../kernel/special_functions/sin_pi.h"
#include "../kernel/special_functions/sin_pi_backward.h"
#include "../kernel/special_functions/sin_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(sin_pi, x)

// sinh(pi * x)
#include "../kernel/special_functions/sinh_pi.h"
#include "../kernel/special_functions/sinh_pi_backward.h"
#include "../kernel/special_functions/sinh_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(sinh_pi, x)

// tan(pi * x)
#include "../kernel/special_functions/tan_pi.h"
#include "../kernel/special_functions/tan_pi_backward.h"
#include "../kernel/special_functions/tan_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(tan_pi, x)

// tanh(pi * x)
#include "../kernel/special_functions/tanh_pi.h"
#include "../kernel/special_functions/tanh_pi_backward.h"
#include "../kernel/special_functions/tanh_pi_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_UNARY_OPERATOR(tanh_pi, x)

// Hahn polynomial Q_n(x; alpha, beta, N)
#include "../kernel/special_functions/hahn_polynomial_q.h"
#include "../kernel/special_functions/hahn_polynomial_q_backward.h"
#include "../kernel/special_functions/hahn_polynomial_q_backward_backward.h"

TORCHSCIENCE_CUDA_POINTWISE_QUINARY_OPERATOR(hahn_polynomial_q, n, x, alpha, beta, N)

// ============================================================================
// Group D: Weierstrass P, Sigma, Zeta (ternary, float+complex)
// ============================================================================
#include "../kernel/special_functions/weierstrass_p.h"
#include "../kernel/special_functions/weierstrass_p_backward.h"
#include "../kernel/special_functions/weierstrass_p_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor weierstrass_p(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(z_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_p",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                    return kernel::special_functions::weierstrass_p(z, g2, g3);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_p_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_z;
    at::Tensor gradient_g2;
    at::Tensor gradient_g3;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_p_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                    -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_p_backward(
                        gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_p_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    if (!gg_z_input.defined() && !gg_g2_input.defined() && !gg_g3_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor grad_grad;
    at::Tensor grad_z;
    at::Tensor grad_g2;
    at::Tensor grad_g3;

    auto z_gg = gg_z_input.defined() ? gg_z_input : at::zeros_like(z_input);
    auto g2_gg = gg_g2_input.defined() ? gg_g2_input : at::zeros_like(g2_input);
    auto g3_gg = gg_g3_input.defined() ? gg_g3_input : at::zeros_like(g3_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_const_input(z_gg)
        .add_const_input(g2_gg)
        .add_const_input(g3_gg)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_p_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t gg_z,
                    scalar_t gg_g2,
                    scalar_t gg_g3,
                    scalar_t gradient,
                    scalar_t z,
                    scalar_t g2,
                    scalar_t g3
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_p_backward_backward(
                        gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("weierstrass_p", torchscience::cuda::special_functions::weierstrass_p);
    module.impl("weierstrass_p_backward", torchscience::cuda::special_functions::weierstrass_p_backward);
    module.impl("weierstrass_p_backward_backward", torchscience::cuda::special_functions::weierstrass_p_backward_backward);
}

#include "../kernel/special_functions/weierstrass_sigma.h"
#include "../kernel/special_functions/weierstrass_sigma_backward.h"
#include "../kernel/special_functions/weierstrass_sigma_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor weierstrass_sigma(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(z_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_sigma",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                    return kernel::special_functions::weierstrass_sigma(z, g2, g3);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_sigma_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_z;
    at::Tensor gradient_g2;
    at::Tensor gradient_g3;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_sigma_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                    -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_sigma_backward(
                        gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_sigma_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    if (!gg_z_input.defined() && !gg_g2_input.defined() && !gg_g3_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor grad_grad;
    at::Tensor grad_z;
    at::Tensor grad_g2;
    at::Tensor grad_g3;

    auto z_gg = gg_z_input.defined() ? gg_z_input : at::zeros_like(z_input);
    auto g2_gg = gg_g2_input.defined() ? gg_g2_input : at::zeros_like(g2_input);
    auto g3_gg = gg_g3_input.defined() ? gg_g3_input : at::zeros_like(g3_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_const_input(z_gg)
        .add_const_input(g2_gg)
        .add_const_input(g3_gg)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_sigma_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t gg_z,
                    scalar_t gg_g2,
                    scalar_t gg_g3,
                    scalar_t gradient,
                    scalar_t z,
                    scalar_t g2,
                    scalar_t g3
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_sigma_backward_backward(
                        gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("weierstrass_sigma", torchscience::cuda::special_functions::weierstrass_sigma);
    module.impl("weierstrass_sigma_backward", torchscience::cuda::special_functions::weierstrass_sigma_backward);
    module.impl("weierstrass_sigma_backward_backward", torchscience::cuda::special_functions::weierstrass_sigma_backward_backward);
}

#include "../kernel/special_functions/weierstrass_zeta.h"
#include "../kernel/special_functions/weierstrass_zeta_backward.h"
#include "../kernel/special_functions/weierstrass_zeta_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor weierstrass_zeta(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(z_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_zeta",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                    return kernel::special_functions::weierstrass_zeta(z, g2, g3);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_zeta_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_z;
    at::Tensor gradient_g2;
    at::Tensor gradient_g3;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_zeta_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                    -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_zeta_backward(
                        gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_zeta_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    if (!gg_z_input.defined() && !gg_g2_input.defined() && !gg_g3_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor grad_grad;
    at::Tensor grad_z;
    at::Tensor grad_g2;
    at::Tensor grad_g3;

    auto z_gg = gg_z_input.defined() ? gg_z_input : at::zeros_like(z_input);
    auto g2_gg = gg_g2_input.defined() ? gg_g2_input : at::zeros_like(g2_input);
    auto g3_gg = gg_g3_input.defined() ? gg_g3_input : at::zeros_like(g3_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_const_input(z_gg)
        .add_const_input(g2_gg)
        .add_const_input(g3_gg)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_zeta_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t gg_z,
                    scalar_t gg_g2,
                    scalar_t gg_g3,
                    scalar_t gradient,
                    scalar_t z,
                    scalar_t g2,
                    scalar_t g3
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_zeta_backward_backward(
                        gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("weierstrass_zeta", torchscience::cuda::special_functions::weierstrass_zeta);
    module.impl("weierstrass_zeta_backward", torchscience::cuda::special_functions::weierstrass_zeta_backward);
    module.impl("weierstrass_zeta_backward_backward", torchscience::cuda::special_functions::weierstrass_zeta_backward_backward);
}

// ============================================================================
// Group C: Weierstrass Eta (binary, float+complex)
// ============================================================================
#include "../kernel/special_functions/weierstrass_eta.h"
#include "../kernel/special_functions/weierstrass_eta_backward.h"
#include "../kernel/special_functions/weierstrass_eta_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor weierstrass_eta(
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(g2_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_eta",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t g2, scalar_t g3) -> scalar_t {
                    return kernel::special_functions::weierstrass_eta(g2, g3);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> weierstrass_eta_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_g2;
    at::Tensor gradient_g3;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_const_input(gradient_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_eta_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t g2, scalar_t g3)
                    -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_eta_backward(
                        gradient, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_eta_backward_backward(
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    if (!gg_g2_input.defined() && !gg_g3_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor grad_grad;
    at::Tensor grad_g2;
    at::Tensor grad_g3;

    auto g2_gg = gg_g2_input.defined() ? gg_g2_input : at::zeros_like(g2_input);
    auto g3_gg = gg_g3_input.defined() ? gg_g3_input : at::zeros_like(g3_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_const_input(g2_gg)
        .add_const_input(g3_gg)
        .add_const_input(gradient_input)
        .add_const_input(g2_input)
        .add_const_input(g3_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "weierstrass_eta_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t gg_g2,
                    scalar_t gg_g3,
                    scalar_t gradient,
                    scalar_t g2,
                    scalar_t g3
                ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::weierstrass_eta_backward_backward(
                        gg_g2, gg_g3, gradient, g2, g3
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("weierstrass_eta", torchscience::cuda::special_functions::weierstrass_eta);
    module.impl("weierstrass_eta_backward", torchscience::cuda::special_functions::weierstrass_eta_backward);
    module.impl("weierstrass_eta_backward_backward", torchscience::cuda::special_functions::weierstrass_eta_backward_backward);
}

// ============================================================================
// Group B: Spherical Hankel functions (binary, complex-only)
// ============================================================================
#include "../kernel/special_functions/spherical_hankel_1.h"
#include "../kernel/special_functions/spherical_hankel_1_backward.h"
#include "../kernel/special_functions/spherical_hankel_1_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor spherical_hankel_1(
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(n_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t n, scalar_t z) -> scalar_t {
                    return kernel::special_functions::spherical_hankel_1(n, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> spherical_hankel_1_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t n, scalar_t z)
                    -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_hankel_1_backward(
                        gradient, n, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> spherical_hankel_1_backward_backward(
    const at::Tensor &n_gradient_gradient_input,
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    if (!n_gradient_gradient_input.defined() && !z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_gradient_output;
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto n_gg = n_gradient_gradient_input.defined()
        ? n_gradient_gradient_input
        : at::zeros_like(n_input);
    auto z_gg = z_gradient_gradient_input.defined()
        ? z_gradient_gradient_input
        : at::zeros_like(z_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(n_gg)
        .add_const_input(z_gg)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t n_gradient_gradient,
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t n,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_hankel_1_backward_backward(
                        n_gradient_gradient, z_gradient_gradient, gradient, n, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("spherical_hankel_1", torchscience::cuda::special_functions::spherical_hankel_1);
    module.impl("spherical_hankel_1_backward", torchscience::cuda::special_functions::spherical_hankel_1_backward);
    module.impl("spherical_hankel_1_backward_backward", torchscience::cuda::special_functions::spherical_hankel_1_backward_backward);
}

#include "../kernel/special_functions/spherical_hankel_2.h"
#include "../kernel/special_functions/spherical_hankel_2_backward.h"
#include "../kernel/special_functions/spherical_hankel_2_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor spherical_hankel_2(
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(n_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t n, scalar_t z) -> scalar_t {
                    return kernel::special_functions::spherical_hankel_2(n, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> spherical_hankel_2_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t n, scalar_t z)
                    -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_hankel_2_backward(
                        gradient, n, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> spherical_hankel_2_backward_backward(
    const at::Tensor &n_gradient_gradient_input,
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    if (!n_gradient_gradient_input.defined() && !z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_gradient_output;
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto n_gg = n_gradient_gradient_input.defined()
        ? n_gradient_gradient_input
        : at::zeros_like(n_input);
    auto z_gg = z_gradient_gradient_input.defined()
        ? z_gradient_gradient_input
        : at::zeros_like(z_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(n_gg)
        .add_const_input(z_gg)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t n_gradient_gradient,
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t n,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_hankel_2_backward_backward(
                        n_gradient_gradient, z_gradient_gradient, gradient, n, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("spherical_hankel_2", torchscience::cuda::special_functions::spherical_hankel_2);
    module.impl("spherical_hankel_2_backward", torchscience::cuda::special_functions::spherical_hankel_2_backward);
    module.impl("spherical_hankel_2_backward_backward", torchscience::cuda::special_functions::spherical_hankel_2_backward_backward);
}

// ============================================================================
// Group F: Spherical Harmonic Y (quaternary, complex with type promotion)
// ============================================================================
#include "../kernel/special_functions/spherical_harmonic_y.h"
#include "../kernel/special_functions/spherical_harmonic_y_backward.h"
#include "../kernel/special_functions/spherical_harmonic_y_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor spherical_harmonic_y(
    const at::Tensor &l_input,
    const at::Tensor &m_input,
    const at::Tensor &theta_input,
    const at::Tensor &phi_input
) {
    c10::cuda::CUDAGuard device_guard(l_input.device());

    auto dtype = at::promote_types(
        at::result_type(l_input, m_input),
        at::result_type(theta_input, phi_input)
    );
    if (!c10::isComplexType(dtype)) {
        dtype = c10::toComplexType(dtype);
    }

    auto l = l_input.to(dtype);
    auto m = m_input.to(dtype);
    auto theta = theta_input.to(dtype);
    auto phi = phi_input.to(dtype);

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(l)
        .add_const_input(m)
        .add_const_input(theta)
        .add_const_input(phi)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_harmonic_y",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t l, scalar_t m, scalar_t theta, scalar_t phi) -> scalar_t {
                    return kernel::special_functions::spherical_harmonic_y(l, m, theta, phi);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
spherical_harmonic_y_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &l_input,
    const at::Tensor &m_input,
    const at::Tensor &theta_input,
    const at::Tensor &phi_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    auto dtype = at::promote_types(
        at::promote_types(
            at::result_type(l_input, m_input),
            at::result_type(theta_input, phi_input)
        ),
        gradient_input.scalar_type()
    );
    if (!c10::isComplexType(dtype)) {
        dtype = c10::toComplexType(dtype);
    }

    auto gradient = gradient_input.to(dtype);
    auto l = l_input.to(dtype);
    auto m = m_input.to(dtype);
    auto theta = theta_input.to(dtype);
    auto phi = phi_input.to(dtype);

    at::Tensor l_gradient_output;
    at::Tensor m_gradient_output;
    at::Tensor theta_gradient_output;
    at::Tensor phi_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(l_gradient_output)
        .add_output(m_gradient_output)
        .add_output(theta_gradient_output)
        .add_output(phi_gradient_output)
        .add_const_input(gradient)
        .add_const_input(l)
        .add_const_input(m)
        .add_const_input(theta)
        .add_const_input(phi)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_harmonic_y_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t l, scalar_t m, scalar_t theta, scalar_t phi)
                    -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_harmonic_y_backward(
                        gradient, l, m, theta, phi
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
spherical_harmonic_y_backward_backward(
    const at::Tensor &l_gradient_gradient_input,
    const at::Tensor &m_gradient_gradient_input,
    const at::Tensor &theta_gradient_gradient_input,
    const at::Tensor &phi_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &l_input,
    const at::Tensor &m_input,
    const at::Tensor &theta_input,
    const at::Tensor &phi_input
) {
    if (!l_gradient_gradient_input.defined() &&
        !m_gradient_gradient_input.defined() &&
        !theta_gradient_gradient_input.defined() &&
        !phi_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    auto dtype = at::promote_types(
        at::promote_types(
            at::result_type(l_input, m_input),
            at::result_type(theta_input, phi_input)
        ),
        gradient_input.scalar_type()
    );
    if (!c10::isComplexType(dtype)) {
        dtype = c10::toComplexType(dtype);
    }

    auto l_gg = l_gradient_gradient_input.defined()
        ? l_gradient_gradient_input.to(dtype)
        : at::zeros_like(l_input, at::TensorOptions().dtype(dtype));
    auto m_gg = m_gradient_gradient_input.defined()
        ? m_gradient_gradient_input.to(dtype)
        : at::zeros_like(m_input, at::TensorOptions().dtype(dtype));
    auto theta_gg = theta_gradient_gradient_input.defined()
        ? theta_gradient_gradient_input.to(dtype)
        : at::zeros_like(theta_input, at::TensorOptions().dtype(dtype));
    auto phi_gg = phi_gradient_gradient_input.defined()
        ? phi_gradient_gradient_input.to(dtype)
        : at::zeros_like(phi_input, at::TensorOptions().dtype(dtype));

    auto gradient = gradient_input.to(dtype);
    auto l = l_input.to(dtype);
    auto m = m_input.to(dtype);
    auto theta = theta_input.to(dtype);
    auto phi = phi_input.to(dtype);

    at::Tensor gradient_gradient_output;
    at::Tensor l_gradient_output;
    at::Tensor m_gradient_output;
    at::Tensor theta_gradient_output;
    at::Tensor phi_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(l_gradient_output)
        .add_output(m_gradient_output)
        .add_output(theta_gradient_output)
        .add_output(phi_gradient_output)
        .add_const_input(l_gg)
        .add_const_input(m_gg)
        .add_const_input(theta_gg)
        .add_const_input(phi_gg)
        .add_const_input(gradient)
        .add_const_input(l)
        .add_const_input(m)
        .add_const_input(theta)
        .add_const_input(phi)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_harmonic_y_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t l_gradient_gradient,
                    scalar_t m_gradient_gradient,
                    scalar_t theta_gradient_gradient,
                    scalar_t phi_gradient_gradient,
                    scalar_t gradient,
                    scalar_t l,
                    scalar_t m,
                    scalar_t theta,
                    scalar_t phi
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::spherical_harmonic_y_backward_backward(
                        l_gradient_gradient, m_gradient_gradient,
                        theta_gradient_gradient, phi_gradient_gradient,
                        gradient, l, m, theta, phi
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result),
                        std::get<4>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2),
            iterator.output(3), iterator.output(4)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("spherical_harmonic_y", torchscience::cuda::special_functions::spherical_harmonic_y);
    module.impl("spherical_harmonic_y_backward", torchscience::cuda::special_functions::spherical_harmonic_y_backward);
    module.impl("spherical_harmonic_y_backward_backward", torchscience::cuda::special_functions::spherical_harmonic_y_backward_backward);
}

// ============================================================================
// Group A: Dawson and Faddeeva W (unary, float+complex)
// ============================================================================
#include "../kernel/special_functions/dawson.h"
#include "../kernel/special_functions/dawson_backward.h"
#include "../kernel/special_functions/dawson_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor dawson(const at::Tensor &z_input) {
    c10::cuda::CUDAGuard device_guard(z_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "dawson",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t z) -> scalar_t {
                    return kernel::special_functions::dawson(z);
                }
            );
        }
    );

    return iterator.output();
}

inline at::Tensor dawson_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "dawson_backward",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t z) -> scalar_t {
                    return kernel::special_functions::dawson_backward(gradient, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> dawson_backward_backward(
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    if (!z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(z_gradient_gradient_input.device());

    at::Tensor gradient_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(z_gradient_gradient_input)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "dawson_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::dawson_backward_backward(
                        z_gradient_gradient, gradient, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("dawson", torchscience::cuda::special_functions::dawson);
    module.impl("dawson_backward", torchscience::cuda::special_functions::dawson_backward);
    module.impl("dawson_backward_backward", torchscience::cuda::special_functions::dawson_backward_backward);
}

#include "../kernel/special_functions/faddeeva_w.h"
#include "../kernel/special_functions/faddeeva_w_backward.h"
#include "../kernel/special_functions/faddeeva_w_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor faddeeva_w(const at::Tensor &z_input) {
    c10::cuda::CUDAGuard device_guard(z_input.device());

    // Promote real inputs to complex
    at::Tensor z_complex;
    if (!z_input.is_complex()) {
        if (z_input.scalar_type() == at::kFloat ||
            z_input.scalar_type() == at::kHalf ||
            z_input.scalar_type() == at::kBFloat16) {
            z_complex = z_input.to(at::kFloat).to(at::kComplexFloat);
        } else {
            z_complex = z_input.to(at::kDouble).to(at::kComplexDouble);
        }
    } else {
        z_complex = z_input;
    }

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_complex)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t z) -> scalar_t {
                    return kernel::special_functions::faddeeva_w(z);
                }
            );
        }
    );

    return iterator.output();
}

inline at::Tensor faddeeva_w_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w_backward",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t z) -> scalar_t {
                    return kernel::special_functions::faddeeva_w_backward(gradient, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> faddeeva_w_backward_backward(
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    if (!z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(z_gradient_gradient_input.device());

    at::Tensor gradient_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(z_gradient_gradient_input)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::faddeeva_w_backward_backward(
                        z_gradient_gradient, gradient, z
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("faddeeva_w", torchscience::cuda::special_functions::faddeeva_w);
    module.impl("faddeeva_w_backward", torchscience::cuda::special_functions::faddeeva_w_backward);
    module.impl("faddeeva_w_backward_backward", torchscience::cuda::special_functions::faddeeva_w_backward_backward);
}

// ============================================================================
// Group E: Voigt Profile (ternary, float-only)
// ============================================================================
#include "../kernel/special_functions/voigt_profile.h"
#include "../kernel/special_functions/voigt_profile_backward.h"
#include "../kernel/special_functions/voigt_profile_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor voigt_profile(
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    c10::cuda::CUDAGuard device_guard(x_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "voigt_profile",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [] GPU_LAMBDA (scalar_t x, scalar_t sigma, scalar_t gamma) -> scalar_t {
                    return kernel::special_functions::voigt_profile(x, sigma, gamma);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> voigt_profile_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor x_gradient_output;
    at::Tensor sigma_gradient_output;
    at::Tensor gamma_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(x_gradient_output)
        .add_output(sigma_gradient_output)
        .add_output(gamma_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "voigt_profile_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (scalar_t gradient, scalar_t x, scalar_t sigma, scalar_t gamma)
                    -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::voigt_profile_backward(
                        gradient, x, sigma, gamma
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voigt_profile_backward_backward(
    const at::Tensor &x_gradient_gradient_input,
    const at::Tensor &sigma_gradient_gradient_input,
    const at::Tensor &gamma_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    if (!x_gradient_gradient_input.defined() &&
        !sigma_gradient_gradient_input.defined() &&
        !gamma_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor gradient_gradient_output;
    at::Tensor x_gradient_output;
    at::Tensor sigma_gradient_output;
    at::Tensor gamma_gradient_output;

    auto x_gg = x_gradient_gradient_input.defined()
        ? x_gradient_gradient_input
        : at::zeros_like(x_input);
    auto sigma_gg = sigma_gradient_gradient_input.defined()
        ? sigma_gradient_gradient_input
        : at::zeros_like(sigma_input);
    auto gamma_gg = gamma_gradient_gradient_input.defined()
        ? gamma_gradient_gradient_input
        : at::zeros_like(gamma_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(x_gradient_output)
        .add_output(sigma_gradient_output)
        .add_output(gamma_gradient_output)
        .add_const_input(x_gg)
        .add_const_input(sigma_gg)
        .add_const_input(gamma_gg)
        .add_const_input(gradient_input)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "voigt_profile_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [] GPU_LAMBDA (
                    scalar_t gg_x,
                    scalar_t gg_sigma,
                    scalar_t gg_gamma,
                    scalar_t gradient,
                    scalar_t x,
                    scalar_t sigma,
                    scalar_t gamma
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    auto result = kernel::special_functions::voigt_profile_backward_backward(
                        gg_x, gg_sigma, gg_gamma, gradient, x, sigma, gamma
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result),
                        std::get<2>(result),
                        std::get<3>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("voigt_profile", torchscience::cuda::special_functions::voigt_profile);
    module.impl("voigt_profile_backward", torchscience::cuda::special_functions::voigt_profile_backward);
    module.impl("voigt_profile_backward_backward", torchscience::cuda::special_functions::voigt_profile_backward_backward);
}

// ============================================================================
// Group G: Log Multivariate Gamma (unary with int64_t param)
// ============================================================================
#include "../kernel/special_functions/log_multivariate_gamma.h"
#include "../kernel/special_functions/log_multivariate_gamma_backward.h"
#include "../kernel/special_functions/log_multivariate_gamma_backward_backward.h"

namespace torchscience::cuda::special_functions {

inline at::Tensor log_multivariate_gamma(const at::Tensor &a_input, int64_t d) {
    c10::cuda::CUDAGuard device_guard(a_input.device());

    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(a_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "log_multivariate_gamma",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [d] GPU_LAMBDA (scalar_t a) -> scalar_t {
                    return kernel::special_functions::log_multivariate_gamma(a, d);
                }
            );
        }
    );

    return iterator.output();
}

inline at::Tensor log_multivariate_gamma_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &a_input,
    int64_t d
) {
    c10::cuda::CUDAGuard device_guard(gradient_input.device());

    at::Tensor grad_a;

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_a)
        .add_const_input(gradient_input)
        .add_const_input(a_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "log_multivariate_gamma_backward",
        [&] {
            at::native::gpu_kernel(
                iterator,
                [d] GPU_LAMBDA (scalar_t gradient, scalar_t a) -> scalar_t {
                    return kernel::special_functions::log_multivariate_gamma_backward(gradient, a, d);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> log_multivariate_gamma_backward_backward(
    const at::Tensor &gg_a_input,
    const at::Tensor &gradient_input,
    const at::Tensor &a_input,
    int64_t d
) {
    if (!gg_a_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    c10::cuda::CUDAGuard device_guard(gg_a_input.device());

    at::Tensor grad_grad_output;
    at::Tensor grad_a;

    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad_output)
        .add_output(grad_a)
        .add_const_input(gg_a_input)
        .add_const_input(gradient_input)
        .add_const_input(a_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        iterator.common_dtype(),
        "log_multivariate_gamma_backward_backward",
        [&] {
            at::native::gpu_kernel_multiple_outputs(
                iterator,
                [d] GPU_LAMBDA (scalar_t gg_a, scalar_t gradient, scalar_t a)
                    -> std::tuple<scalar_t, scalar_t> {
                    auto result = kernel::special_functions::log_multivariate_gamma_backward_backward(
                        gg_a, gradient, a, d
                    );
                    return std::make_tuple(
                        std::get<0>(result),
                        std::get<1>(result)
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

} // namespace torchscience::cuda::special_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("log_multivariate_gamma", torchscience::cuda::special_functions::log_multivariate_gamma);
    module.impl("log_multivariate_gamma_backward", torchscience::cuda::special_functions::log_multivariate_gamma_backward);
    module.impl("log_multivariate_gamma_backward_backward", torchscience::cuda::special_functions::log_multivariate_gamma_backward_backward);
}
