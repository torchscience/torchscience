#pragma once

/*
 * Log-Gamma Function log(Gamma(z))
 *
 * DESIGN NOTES:
 *
 * 1. MATHEMATICAL DEFINITION:
 *    log(Gamma(z)) is the natural logarithm of the gamma function.
 *    It is defined for all complex z except non-positive integers.
 *
 * 2. IMPLEMENTATION:
 *    Uses Lanczos approximation in log form to avoid overflow/underflow:
 *      log(Gamma(z)) = 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g(z))
 *    where t = z + g - 0.5 and A_g is the Lanczos series.
 *
 *    For Re(z) < 0.5, uses the reflection formula:
 *      log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
 *
 * 3. SPECIAL VALUES:
 *    - Returns infinity at poles (non-positive integers)
 *
 * 4. APPLICATIONS:
 *    - Hypergeometric functions (avoids overflow in Gamma ratios)
 *    - Incomplete beta function
 *    - Statistical distributions
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

#include "lanczos_approximation.h"  // for lanczos_series, kLanczosG
#include "sin_pi.h"                 // for sin_pi
#include "digamma.h"                // for is_nonpositive_integer

namespace torchscience::impl::special_functions {

// Pi constant for log_gamma
constexpr double kPi_LogGamma = 3.14159265358979323846264338327950288;

/**
 * Log of the gamma function for complex arguments: log(Gamma(z))
 *
 * Uses Lanczos approximation in log form to avoid overflow/underflow:
 *   log(Gamma(z)) = 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g(z))
 *   where t = z + g - 0.5 and A_g is the Lanczos series.
 *
 * For Re(z) < 0.5, uses the reflection formula:
 *   log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
 *
 * Returns infinity at poles (non-positive integers).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> log_gamma_complex(c10::complex<T> z) {
  using std::log;
  using std::abs;

  const T pi_val = T(kPi_LogGamma);
  const T g = T(kLanczosG);

  // Helper to create real-valued complex constants
  const auto real = [](T val) { return c10::complex<T>(val, T(0)); };

  // Check for poles at non-positive integers
  if (is_nonpositive_integer(z)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // For Re(z) < 0.5, use reflection formula:
  // log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
  if (z.real() < T(0.5)) {
    auto sin_pi_z = sin_pi(z);
    // Handle case where sin(pi*z) is zero (shouldn't happen if not at pole)
    if (abs(sin_pi_z) < T(1e-300)) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
    return real(log(pi_val)) - log(sin_pi_z) -
           log_gamma_complex(real(T(1)) - z);
  }

  // Lanczos approximation in log form for Re(z) >= 0.5
  // log(Gamma(z)) = 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g(z))
  // where t = z + g - 0.5
  auto A_g = lanczos_series(z);  // From gamma.h
  auto t = z + real(g - T(0.5));

  // Compute in log form: 0.5*log(2*pi) + (z-0.5)*log(t) - t + log(A_g)
  return real(T(0.5) * log(T(2) * pi_val)) +
         (z - real(T(0.5))) * log(t) - t + log(A_g);
}

}  // namespace torchscience::impl::special_functions
