#pragma once

#include "../impl/special_functions/gamma_traits.h"
#include "../impl/special_functions/chebyshev_polynomial_t_traits.h"
#include "../impl/special_functions/incomplete_beta_traits.h"
#include "../impl/special_functions/hypergeometric_2_f_1_traits.h"

#include "operators.h"

// Template-based registration using ImplTraits
using torchscience::impl::special_functions::GammaImpl;
using torchscience::impl::special_functions::ChebyshevPolynomialTImpl;
using torchscience::impl::special_functions::IncompleteBetaImpl;
using torchscience::impl::special_functions::Hypergeometric2F1Impl;

TORCH_LIBRARY_IMPL(torchscience, Autograd, m_autograd_special_functions) {
    REGISTER_AUTOGRAD_UNARY(m_autograd_special_functions, gamma, GammaImpl);
    REGISTER_AUTOGRAD_BINARY(m_autograd_special_functions, chebyshev_polynomial_t, ChebyshevPolynomialTImpl);
    REGISTER_AUTOGRAD_TERNARY(m_autograd_special_functions, incomplete_beta, IncompleteBetaImpl);
    REGISTER_AUTOGRAD_QUATERNARY(m_autograd_special_functions, hypergeometric_2_f_1, Hypergeometric2F1Impl);
}
