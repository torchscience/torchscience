#pragma once

#include "macros.h"

TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(gamma, Gamma, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(digamma, Digamma, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(trigamma, Trigamma, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(beta, Beta, a, b)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, ChebyshevPolynomialT, x, n)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_u, ChebyshevPolynomialU, x, n)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(incomplete_beta, IncompleteBeta, x, a, b)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_2_f_1, Hypergeometric2F1, a, b, c, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_m, ConfluentHypergeometricM, a, b, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_u, ConfluentHypergeometricU, a, b, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(polygamma, Polygamma, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(log_beta, LogBeta, a, b)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(log_gamma, LogGamma, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(reciprocal_gamma, ReciprocalGamma, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(gamma_sign, GammaSign, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(regularized_gamma_p, RegularizedGammaP, a, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(regularized_gamma_q, RegularizedGammaQ, a, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(modified_bessel_i_0, ModifiedBesselI0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(modified_bessel_i_1, ModifiedBesselI1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(bessel_j_0, BesselJ0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(bessel_j_1, BesselJ1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(bessel_y_0, BesselY0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(bessel_y_1, BesselY1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(modified_bessel_k_0, ModifiedBesselK0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(modified_bessel_k_1, ModifiedBesselK1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(bessel_j, BesselJ, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(bessel_y, BesselY, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(modified_bessel_k, ModifiedBesselK, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(modified_bessel_i, ModifiedBesselI, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_f, CarlsonEllipticIntegralRF, x, y, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_d, CarlsonEllipticIntegralRD, x, y, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_c, CarlsonEllipticIntegralRC, x, y)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(carlson_elliptic_integral_r_j, CarlsonEllipticIntegralRJ, x, y, z, p)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_g, CarlsonEllipticIntegralRG, x, y, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_e, CarlsonEllipticIntegralRE, x, y, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_m, CarlsonEllipticIntegralRM, x, y, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_k, CarlsonEllipticIntegralRK, x, y)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_k, CompleteLegendreEllipticIntegralK, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_e, CompleteLegendreEllipticIntegralE, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_e, IncompleteLegendreEllipticIntegralE, phi, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_f, IncompleteLegendreEllipticIntegralF, phi, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(complete_legendre_elliptic_integral_pi, CompleteLegendreEllipticIntegralPi, n, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(incomplete_legendre_elliptic_integral_pi, IncompleteLegendreEllipticIntegralPi, n, phi, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_amplitude_am, JacobiAmplitudeAm, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dn, JacobiEllipticDn, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cn, JacobiEllipticCn, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sn, JacobiEllipticSn, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sd, JacobiEllipticSd, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cd, JacobiEllipticCd, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sc, JacobiEllipticSc, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nd, JacobiEllipticNd, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nc, JacobiEllipticNc, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ns, JacobiEllipticNs, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dc, JacobiEllipticDc, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ds, JacobiEllipticDs, u, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cs, JacobiEllipticCs, u, m)

// Inverse Jacobi elliptic functions (primary)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sn, InverseJacobiEllipticSn, x, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cn, InverseJacobiEllipticCn, x, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_dn, InverseJacobiEllipticDn, x, m)

// Inverse Jacobi elliptic functions (derived)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sd, InverseJacobiEllipticSd, x, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cd, InverseJacobiEllipticCd, x, m)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sc, InverseJacobiEllipticSc, x, m)

// Jacobi theta functions
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(theta_1, Theta1, z, q)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(theta_2, Theta2, z, q)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(theta_3, Theta3, z, q)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(theta_4, Theta4, z, q)

// Weierstrass elliptic function P
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(weierstrass_p, WeierstrassP, z, g2, g3)

// Weierstrass sigma function
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(weierstrass_sigma, WeierstrassSigma, z, g2, g3)

// Weierstrass zeta function
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(weierstrass_zeta, WeierstrassZeta, z, g2, g3)

// Weierstrass eta quasi-period
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(weierstrass_eta, WeierstrassEta, g2, g3)

TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_0, SphericalBesselJ0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_1, SphericalBesselJ1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_bessel_j, SphericalBesselJ, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_0, SphericalBesselY0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_1, SphericalBesselY1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_bessel_y, SphericalBesselY, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_0, SphericalBesselI0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_1, SphericalBesselI1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_bessel_i, SphericalBesselI, n, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_0, SphericalBesselK0, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_1, SphericalBesselK1, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_bessel_k, SphericalBesselK, n, z)

// Exponential integrals
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(exponential_integral_ei, ExponentialIntegralEi, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(exponential_integral_e_1, ExponentialIntegralE1, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(exponential_integral_ein, ExponentialIntegralEin, x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(exponential_integral_e, ExponentialIntegralE, n, x)

// Sine integral
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(sine_integral_si, SineIntegralSi, x)

// Cosine integral
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(cosine_integral_ci, CosineIntegralCi, x)

// Spherical Hankel functions of the first kind
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_hankel_1, SphericalHankel1, n, z)

// Spherical Hankel functions of the second kind
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(spherical_hankel_2, SphericalHankel2, n, z)

// Spherical harmonic Y_l^m(theta, phi)
// Custom autograd implementation: promotes real inputs to complex before saving,
// since the forward always produces complex output.
namespace torchscience::autograd::special_functions {

inline at::ScalarType spherical_harmonic_y_complex_dtype(
    const at::Tensor &l, const at::Tensor &m,
    const at::Tensor &theta, const at::Tensor &phi
) {
    auto dtype = at::promote_types(
        at::result_type(l, m),
        at::result_type(theta, phi)
    );
    if (!c10::isComplexType(dtype)) {
        dtype = c10::toComplexType(dtype);
    }
    return dtype;
}

class SphericalHarmonicYBackward : public torch::autograd::Function<SphericalHarmonicYBackward> {
public:
  static std::vector<at::Tensor> forward(
    torch::autograd::AutogradContext *context,
    const at::Tensor &gradient_output,
    const at::Tensor &l_input,
    const at::Tensor &m_input,
    const at::Tensor &theta_input,
    const at::Tensor &phi_input,
    bool l_requires_gradient,
    bool m_requires_gradient,
    bool theta_requires_gradient,
    bool phi_requires_gradient
  ) {
    context->save_for_backward(
      {gradient_output, l_input, m_input, theta_input, phi_input}
    );

    context->saved_data["l_requires_grad"] = l_requires_gradient;
    context->saved_data["m_requires_grad"] = m_requires_gradient;
    context->saved_data["theta_requires_grad"] = theta_requires_gradient;
    context->saved_data["phi_requires_grad"] = phi_requires_gradient;

    at::AutoDispatchBelowAutograd guard;

    static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::spherical_harmonic_y_backward", "")
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        const at::Tensor &, const at::Tensor &, const at::Tensor &,
        const at::Tensor &, const at::Tensor &
      )>();

    auto [l_gradient, m_gradient, theta_gradient, phi_gradient] = op.call(
      gradient_output, l_input, m_input, theta_input, phi_input
    );

    return {l_gradient, m_gradient, theta_gradient, phi_gradient};
  }

  static std::vector<at::Tensor> backward(
    torch::autograd::AutogradContext *context,
    const std::vector<at::Tensor> &gradient_outputs
  ) {
    torch::autograd::variable_list saved_variables = context->get_saved_variables();

    bool l_requires_grad = context->saved_data["l_requires_grad"].toBool();
    bool m_requires_grad = context->saved_data["m_requires_grad"].toBool();
    bool theta_requires_grad = context->saved_data["theta_requires_grad"].toBool();
    bool phi_requires_grad = context->saved_data["phi_requires_grad"].toBool();

    if ((!gradient_outputs[0].defined() &&
         !gradient_outputs[1].defined() &&
         !gradient_outputs[2].defined() &&
         !gradient_outputs[3].defined()) ||
        (!l_requires_grad && !m_requires_grad &&
         !theta_requires_grad && !phi_requires_grad)) {
      return {
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
      };
    }

    at::AutoDispatchBelowAutograd guard;

    auto l_gg = gradient_outputs[0].defined()
      ? gradient_outputs[0] : at::zeros_like(saved_variables[1]);
    auto m_gg = gradient_outputs[1].defined()
      ? gradient_outputs[1] : at::zeros_like(saved_variables[2]);
    auto theta_gg = gradient_outputs[2].defined()
      ? gradient_outputs[2] : at::zeros_like(saved_variables[3]);
    auto phi_gg = gradient_outputs[3].defined()
      ? gradient_outputs[3] : at::zeros_like(saved_variables[4]);

    auto [
      gradient_gradient_output,
      l_gradient_output,
      m_gradient_output,
      theta_gradient_output,
      phi_gradient_output
    ] = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::spherical_harmonic_y_backward_backward", "")
      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
        const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
        const at::Tensor &
      )>()
      .call(
        l_gg, m_gg, theta_gg, phi_gg,
        saved_variables[0], saved_variables[1], saved_variables[2],
        saved_variables[3], saved_variables[4]
      );

    return {
      gradient_gradient_output,
      l_gradient_output, m_gradient_output,
      theta_gradient_output, phi_gradient_output,
      at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
    };
  }
};

class SphericalHarmonicY : public torch::autograd::Function<SphericalHarmonicY> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext *context,
    const at::Tensor &l_input,
    const at::Tensor &m_input,
    const at::Tensor &theta_input,
    const at::Tensor &phi_input
  ) {
    // Promote to complex before saving for backward
    auto dtype = spherical_harmonic_y_complex_dtype(l_input, m_input, theta_input, phi_input);
    auto l = l_input.to(dtype);
    auto m = m_input.to(dtype);
    auto theta = theta_input.to(dtype);
    auto phi = phi_input.to(dtype);

    context->save_for_backward({l, m, theta, phi});

    context->saved_data["l_requires_grad"] = l_input.requires_grad() && (at::isFloatingType(l_input.scalar_type()) || at::isComplexType(l_input.scalar_type()));
    context->saved_data["m_requires_grad"] = m_input.requires_grad() && (at::isFloatingType(m_input.scalar_type()) || at::isComplexType(m_input.scalar_type()));
    context->saved_data["theta_requires_grad"] = theta_input.requires_grad() && (at::isFloatingType(theta_input.scalar_type()) || at::isComplexType(theta_input.scalar_type()));
    context->saved_data["phi_requires_grad"] = phi_input.requires_grad() && (at::isFloatingType(phi_input.scalar_type()) || at::isComplexType(phi_input.scalar_type()));

    // Save original input dtypes so backward can convert gradients appropriately
    context->saved_data["l_is_complex"] = at::isComplexType(l_input.scalar_type());
    context->saved_data["m_is_complex"] = at::isComplexType(m_input.scalar_type());
    context->saved_data["theta_is_complex"] = at::isComplexType(theta_input.scalar_type());
    context->saved_data["phi_is_complex"] = at::isComplexType(phi_input.scalar_type());

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchscience::spherical_harmonic_y", "")
      .typed<at::Tensor(
        const at::Tensor &, const at::Tensor &,
        const at::Tensor &, const at::Tensor &
      )>()
      .call(l_input, m_input, theta_input, phi_input);
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext *context,
    const torch::autograd::variable_list &gradient_outputs
  ) {
    torch::autograd::variable_list variables = context->get_saved_variables();

    bool l_requires_grad = context->saved_data["l_requires_grad"].toBool();
    bool m_requires_grad = context->saved_data["m_requires_grad"].toBool();
    bool theta_requires_grad = context->saved_data["theta_requires_grad"].toBool();
    bool phi_requires_grad = context->saved_data["phi_requires_grad"].toBool();

    bool l_is_complex = context->saved_data["l_is_complex"].toBool();
    bool m_is_complex = context->saved_data["m_is_complex"].toBool();
    bool theta_is_complex = context->saved_data["theta_is_complex"].toBool();
    bool phi_is_complex = context->saved_data["phi_is_complex"].toBool();

    auto gradients = SphericalHarmonicYBackward::apply(
      gradient_outputs[0],
      variables[0], variables[1], variables[2], variables[3],
      l_requires_grad, m_requires_grad, theta_requires_grad, phi_requires_grad
    );

    // Project complex gradients back to real for originally-real inputs
    auto project = [](const at::Tensor &grad, bool was_complex) -> at::Tensor {
      if (was_complex || !grad.is_complex()) return grad;
      return at::real(grad);
    };

    return {
      l_requires_grad ? project(gradients[0], l_is_complex) : at::Tensor(),
      m_requires_grad ? project(gradients[1], m_is_complex) : at::Tensor(),
      theta_requires_grad ? project(gradients[2], theta_is_complex) : at::Tensor(),
      phi_requires_grad ? project(gradients[3], phi_is_complex) : at::Tensor()
    };
  }
};

inline at::Tensor spherical_harmonic_y(
  const at::Tensor &l, const at::Tensor &m,
  const at::Tensor &theta, const at::Tensor &phi
) {
  return SphericalHarmonicY::apply(l, m, theta, phi);
}

} // namespace torchscience::autograd::special_functions

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
  module.impl("spherical_harmonic_y", torchscience::autograd::special_functions::spherical_harmonic_y);
}

// Airy function of the first kind
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(airy_ai, AiryAi, x)

// Airy function of the second kind
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(airy_bi, AiryBi, x)

// Lambert W function (product logarithm)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(lambert_w, LambertW, k, z)

// Kelvin function ber (real part of J_0 at rotated argument)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(kelvin_ber, KelvinBer, x)

// Kelvin function bei (imaginary part of J_0 at rotated argument)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(kelvin_bei, KelvinBei, x)

// Kelvin function ker (real part of K_0 at rotated argument)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(kelvin_ker, KelvinKer, x)

// Kelvin function kei (imaginary part of K_0 at rotated argument)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(kelvin_kei, KelvinKei, x)

// Riemann zeta function (s > 1 only)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(zeta, Zeta, s)

// Polylogarithm function Li_s(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(polylogarithm_li, PolylogarithmLi, s, z)

TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_u, ParabolicCylinderU, a, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_v, ParabolicCylinderV, a, z)

// Whittaker functions
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(whittaker_m, WhittakerM, kappa, mu, z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(whittaker_w, WhittakerW, kappa, mu, z)

// Hypergeometric 0F1 (confluent hypergeometric limit function)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(hypergeometric_0_f_1, Hypergeometric0F1, b, z)

// Hypergeometric 1F2
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_1_f_2, Hypergeometric1F2, a, b1, b2, z)

// Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(faddeeva_w, FaddeevaW, z)

// Inverse error function
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(erfinv, Erfinv, x)

// Inverse complementary error function
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(erfcinv, Erfcinv, x)

// Fresnel sine integral
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(fresnel_s, FresnelS, z)

// Fresnel cosine integral
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(fresnel_c, FresnelC, z)

// Dawson's integral
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(dawson, Dawson, z)

// Voigt profile
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(voigt_profile, VoigtProfile, x, sigma, gamma)

// Generalized hypergeometric pFq - custom autograd implementation
namespace torchscience::autograd::special_functions {

struct HypergeometricPFQFunction : public torch::autograd::Function<HypergeometricPFQFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &a,
        const at::Tensor &b,
        const at::Tensor &z
    ) {
        ctx->save_for_backward({a, b, z});
        // Exclude Autograd keys to prevent re-dispatching to this function
        c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::hypergeometric_p_f_q", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(a, b, z);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<at::Tensor> grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto a = saved[0];
        auto b = saved[1];
        auto z = saved[2];

        // Exclude Autograd keys for the backward call
        c10::impl::ExcludeDispatchKeyGuard no_autograd(c10::autograd_dispatch_keyset);
        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::hypergeometric_p_f_q_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(grad_outputs[0], a, b, z);

        return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
    }
};

inline at::Tensor hypergeometric_p_f_q_autograd(
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &z
) {
    return HypergeometricPFQFunction::apply(a, b, z);
}

} // namespace torchscience::autograd::special_functions

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("hypergeometric_p_f_q", torchscience::autograd::special_functions::hypergeometric_p_f_q_autograd);
}

// Legendre polynomial P_n(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(legendre_polynomial_p, LegendrePolynomialP, n, z)

// Legendre function of the second kind Q_n(x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(legendre_polynomial_q, LegendrePolynomialQ, x, n)

// Associated Legendre polynomial P_n^m(x)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(associated_legendre_polynomial_p, AssociatedLegendrePolynomialP, n, m, x)

// Hermite polynomial (physicists') H_n(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(hermite_polynomial_h, HermitePolynomialH, n, z)

// Hermite polynomial (probabilists') He_n(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(hermite_polynomial_he, HermitePolynomialHe, n, z)

// Generalized Laguerre polynomial L_n^alpha(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(laguerre_polynomial_l, LaguerrePolynomialL, n, alpha, z)

// Gegenbauer (ultraspherical) polynomial C_n^lambda(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(gegenbauer_polynomial_c, GegenbauerPolynomialC, n, lambda, z)

// Jacobi polynomial P_n^(alpha,beta)(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(jacobi_polynomial_p, JacobiPolynomialP, n, alpha, beta, z)

// Radial Zernike polynomial R_n^m(rho)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(zernike_polynomial_r, ZernikePolynomialR, n, m, rho)

// Full Zernike polynomial Z_n^m(rho, theta)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(zernike_polynomial_z, ZernikePolynomialZ, n, m, rho, theta)

// Krawtchouk polynomial K_n(x; p, N)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(krawtchouk_polynomial_k, KrawtchoukPolynomialK, n, x, p, N)

// Meixner polynomial M_n(x; beta, c)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR(meixner_polynomial_m, MeixnerPolynomialM, n, x, beta, c)

// Hahn polynomial Q_n(x; alpha, beta, N)
TORCHSCIENCE_AUTOGRAD_POINTWISE_QUINARY_OPERATOR(hahn_polynomial_q, HahnPolynomialQ, n, x, alpha, beta, N)

// Charlier polynomial C_n(x; a)
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(charlier_polynomial_c, CharlierPolynomialC, n, x, a)

// Pochhammer symbol (rising factorial)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(pochhammer, Pochhammer, z, m)

// Log multivariate gamma autograd - custom because d is int
namespace torchscience::autograd::special_functions {

// Backward function class - enables second-order gradients
class LogMultivariateGammaBackward : public torch::autograd::Function<LogMultivariateGammaBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &grad_output,
        const at::Tensor &a,
        int64_t d,
        bool a_requires_grad
    ) {
        ctx->save_for_backward({grad_output, a});
        ctx->saved_data["d"] = d;
        ctx->saved_data["a_requires_grad"] = a_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto grad_a = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::log_multivariate_gamma_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>()
            .call(grad_output, a, d);

        return {grad_a};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        const std::vector<at::Tensor> &grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto grad_output = saved[0];
        auto a = saved[1];
        auto d = ctx->saved_data["d"].toInt();
        bool a_requires_grad = ctx->saved_data["a_requires_grad"].toBool();

        if (!grad_outputs[0].defined() || !a_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad_a] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::log_multivariate_gamma_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t)>()
            .call(grad_outputs[0], grad_output, a, d);

        return {grad_grad_output, grad_a, at::Tensor(), at::Tensor()};
    }
};

// Main forward function class
class LogMultivariateGamma : public torch::autograd::Function<LogMultivariateGamma> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &a,
        int64_t d
    ) {
        ctx->save_for_backward({a});
        ctx->saved_data["d"] = d;
        ctx->saved_data["a_requires_grad"] = a.requires_grad() && (at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()));

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::log_multivariate_gamma", "")
            .typed<at::Tensor(const at::Tensor&, int64_t)>()
            .call(a, d);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        const torch::autograd::variable_list &grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto a = saved[0];
        auto d = ctx->saved_data["d"].toInt();
        bool a_requires_grad = ctx->saved_data["a_requires_grad"].toBool();

        auto gradients = LogMultivariateGammaBackward::apply(
            grad_outputs[0],
            a,
            d,
            a_requires_grad
        );

        at::Tensor grad_a;
        if (a_requires_grad) {
            grad_a = gradients[0];
        } else {
            grad_a = at::Tensor();
        }

        return {grad_a, at::Tensor()};  // No gradient for d
    }
};

inline at::Tensor log_multivariate_gamma(const at::Tensor &a, int64_t d) {
    return LogMultivariateGamma::apply(a, d);
}

} // namespace torchscience::autograd::special_functions

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("log_multivariate_gamma", torchscience::autograd::special_functions::log_multivariate_gamma);
}

// Inverse regularized gamma P function
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_p, InverseRegularizedGammaP, a, y)

// Inverse regularized gamma Q function
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_q, InverseRegularizedGammaQ, a, y)

// Inverse regularized incomplete beta function
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(inverse_regularized_incomplete_beta, InverseRegularizedIncompleteBeta, a, b, y)

// Inverse complementary regularized incomplete beta function
TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR(inverse_complementary_regularized_incomplete_beta, InverseComplementaryRegularizedIncompleteBeta, a, b, y)
// Struve function H_0
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(struve_h_0, StruveH0, z)

// Struve function H_1
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(struve_h_1, StruveH1, z)

// Modified Struve function L_0
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(struve_l_0, StruveL0, z)

// Modified Struve function L_1
TORCHSCIENCE_AUTOGRAD_POINTWISE_UNARY_OPERATOR(struve_l_1, StruveL1, z)

// General order Struve function H_n(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(struve_h, StruveH, n, z)

// General order modified Struve function L_n(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(struve_l, StruveL, n, z)

// Anger function J_nu(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(anger_j, AngerJ, n, z)

// Weber function E_nu(z)
TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(weber_e, WeberE, n, z)
