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
