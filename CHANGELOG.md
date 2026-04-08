# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-08

### Added

- `torchscience.special_functions` module with 152 operators
- Full backend support: CPU, CUDA, Meta, Autograd, Autocast, Sparse (COO/CSR), Quantized
- First and second-order automatic differentiation for all operators
- torch.compile compatibility for all operators
- Autocast (mixed precision) support for all operators
- vmap support via FuncTorch batching rules
- Complex tensor support for operators with complex-valued kernels
- `torchscience.testing` module with OpTestCase framework

### Operators

Gamma and related: gamma, digamma, trigamma, tetragamma, pentagamma, polygamma,
log_gamma, reciprocal_gamma, gamma_sign, regularized_gamma_p, regularized_gamma_q,
inverse_regularized_gamma_p, inverse_regularized_gamma_q, log_multivariate_gamma

Beta and related: beta, log_beta, incomplete_beta, inverse_regularized_incomplete_beta,
inverse_complementary_regularized_incomplete_beta

Hypergeometric: hypergeometric_0_f_1, hypergeometric_1_f_2, hypergeometric_2_f_1,
hypergeometric_p_f_q, confluent_hypergeometric_m, confluent_hypergeometric_u

Bessel: bessel_j_0, bessel_j_1, bessel_j, bessel_y_0, bessel_y_1, bessel_y,
modified_bessel_i_0, modified_bessel_i_1, modified_bessel_i,
modified_bessel_k_0, modified_bessel_k_1, modified_bessel_k

Spherical Bessel: spherical_bessel_j_0, spherical_bessel_j_1, spherical_bessel_j,
spherical_bessel_y_0, spherical_bessel_y_1, spherical_bessel_y,
spherical_bessel_i_0, spherical_bessel_i_1, spherical_bessel_i,
spherical_bessel_k_0, spherical_bessel_k_1, spherical_bessel_k,
spherical_hankel_1, spherical_hankel_2

Elliptic integrals: complete_legendre_elliptic_integral_k,
complete_legendre_elliptic_integral_e, complete_legendre_elliptic_integral_pi,
incomplete_legendre_elliptic_integral_f, incomplete_legendre_elliptic_integral_e,
incomplete_legendre_elliptic_integral_pi, carlson_elliptic_integral_r_f,
carlson_elliptic_integral_r_d, carlson_elliptic_integral_r_c,
carlson_elliptic_integral_r_j, carlson_elliptic_integral_r_g,
carlson_elliptic_integral_r_e, carlson_elliptic_integral_r_m,
carlson_elliptic_integral_r_k

Jacobi elliptic: jacobi_elliptic_sn, jacobi_elliptic_cn, jacobi_elliptic_dn,
jacobi_elliptic_sd, jacobi_elliptic_cd, jacobi_elliptic_sc,
jacobi_elliptic_nd, jacobi_elliptic_nc, jacobi_elliptic_ns,
jacobi_elliptic_dc, jacobi_elliptic_ds, jacobi_elliptic_cs,
jacobi_amplitude_am, inverse_jacobi_elliptic_sn, inverse_jacobi_elliptic_cn,
inverse_jacobi_elliptic_dn, inverse_jacobi_elliptic_sd,
inverse_jacobi_elliptic_cd, inverse_jacobi_elliptic_sc

Theta functions: theta_1, theta_2, theta_3, theta_4

Weierstrass: weierstrass_p, weierstrass_sigma, weierstrass_zeta, weierstrass_eta

Orthogonal polynomials: chebyshev_polynomial_t, chebyshev_polynomial_u,
chebyshev_polynomial_v, chebyshev_polynomial_w, legendre_polynomial_p,
legendre_polynomial_q, associated_legendre_polynomial_p, hermite_polynomial_h,
hermite_polynomial_he, laguerre_polynomial_l, gegenbauer_polynomial_c,
jacobi_polynomial_p, zernike_polynomial_r, zernike_polynomial_z

Discrete orthogonal polynomials: krawtchouk_polynomial_k, meixner_polynomial_m,
hahn_polynomial_q, charlier_polynomial_c

Trigonometric-pi: sin_pi, cos_pi, tan_pi, sinh_pi, cosh_pi, tanh_pi

Airy: airy_ai, airy_bi

Error functions: erfinv, erfcinv, faddeeva_w, dawson

Exponential integrals: exponential_integral_ei, exponential_integral_e_1,
exponential_integral_ein, exponential_integral_e

Other: fresnel_s, fresnel_c, sine_integral_si, cosine_integral_ci,
kelvin_ber, kelvin_bei, kelvin_ker, kelvin_kei, struve_h_0, struve_h_1,
struve_h, struve_l_0, struve_l_1, struve_l, anger_j, weber_e,
lambert_w, pochhammer, zeta, polylogarithm_li, voigt_profile,
parabolic_cylinder_u, parabolic_cylinder_v, whittaker_m, whittaker_w,
spherical_harmonic_y
