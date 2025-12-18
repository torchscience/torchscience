#pragma once

/*
 * Lanczos Approximation for the Gamma Function
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL BACKGROUND:
 *    The Lanczos approximation provides a way to compute the gamma function
 *    with high precision using a rational function approximation:
 *
 *    Γ(z+1) = √(2π) * (z + g + 0.5)^(z + 0.5) * e^(-(z + g + 0.5)) * A_g(z)
 *
 *    where g is a parameter that controls the tradeoff between the number
 *    of terms and the accuracy, and A_g(z) is a series approximation.
 *
 * 2. PARAMETERS:
 *    - g = 7: Lanczos parameter
 *    - n = 9: Number of coefficients
 *    - Provides ~15 digits of precision for double
 *
 * 3. USAGE:
 *    - Used by gamma.h for Γ(z) computation
 *    - Used by log_gamma.h for log(Γ(z)) computation
 *
 * 4. REFERENCES:
 *    - Lanczos, C. (1964). "A Precision Approximation of the Gamma Function"
 *    - SIAM Journal on Numerical Analysis, Series B, Vol. 1, pp. 86-96
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <type_traits>

namespace torchscience::impl::special_functions {

// ============================================================================
// Lanczos coefficients (g=7, n=9)
// ============================================================================

// Lanczos approximation parameter
constexpr double kLanczosG = 7.0;

// Number of Lanczos coefficients
constexpr int kLanczosN = 9;

// Lanczos coefficients for g=7, n=9
// These provide ~15 digits of precision for double
constexpr double kLanczosCoeffs[kLanczosN] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
  -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
  -176.61502916214059906584551354,
  12.507343278686904814458936853,
  -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};

// sqrt(2 * pi)
constexpr double kSqrt2Pi = 2.5066282746310005024157652848110452530069867406099;

// ============================================================================
// Lanczos series computation
// ============================================================================

/**
 * Compute the Lanczos series A_g(z) for real types.
 *
 * A_g(z) = c_0 + sum_{k=1}^{n-1} c_k / (z + k - 1)
 *
 * This is the rational function part of the Lanczos approximation.
 */
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<
  !c10::is_complex<scalar_t>::value &&
  (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>),
  scalar_t>
lanczos_series(scalar_t z) {
  scalar_t output = scalar_t(kLanczosCoeffs[0]);

  for (int index = 1; index < kLanczosN; index++) {
    output = output + (scalar_t(kLanczosCoeffs[index]) / (z + scalar_t(index - 1)));
  }

  return output;
}

/**
 * Compute the Lanczos series A_g(z) for complex types.
 *
 * A_g(z) = c_0 + sum_{k=1}^{n-1} c_k / (z + k - 1)
 *
 * This is the rational function part of the Lanczos approximation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> lanczos_series(c10::complex<T> z) {
  const auto real = [](T val) {
    return c10::complex<T>(val, T(0));
  };

  c10::complex<T> output = real(T(kLanczosCoeffs[0]));

  for (int index = 1; index < kLanczosN; index++) {
    output = output + real(T(kLanczosCoeffs[index])) / (z + real(T(index - 1)));
  }

  return output;
}

}  // namespace torchscience::impl::special_functions
