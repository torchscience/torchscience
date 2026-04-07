#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Spherical harmonic Y_l^m(theta, phi)
//
// Mathematical Definition:
//   Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * exp(i*m*phi)
//
// where:
//   N_l^m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) is the normalization factor
//   P_l^m is the associated Legendre polynomial (Condon-Shortley phase included)
//
// Associated Legendre polynomial via recursion:
//   1. Seed: P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
//   2. Seed: P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
//   3. Recurrence: (l-m)*P_l^m(x) = x*(2l-1)*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)
//
// Negative m handled via:
//   Y_l^{-|m|} = (-1)^m * conj(Y_l^{|m|})
//
// Special Values:
//   Y_0^0 = 1/sqrt(4*pi) for all (theta, phi)
//   Y_l^0 = sqrt((2l+1)/(4*pi)) * P_l(cos(theta))
//
// Applications:
//   - Quantum mechanics (angular momentum eigenfunctions)
//   - Electrostatics (multipole expansion)
//   - Computer graphics (environment lighting, PRT)
//   - Geophysics (gravitational/magnetic field modeling)

namespace detail {

// Compute associated Legendre polynomial P_l^m(x) using recursion.
// Includes Condon-Shortley phase (-1)^m.
// Assumes m >= 0, l >= m.
template <typename T>
T associated_legendre_p(int l, int m, T x) {
  // Seed: P_m^m(x) = (-1)^m * (2m-1)!! * (1 - x^2)^(m/2)
  T pmm = T(1);
  if (m > 0) {
    T somx2 = std::sqrt(T(1) - x * x);
    T fact = T(1);
    for (int i = 1; i <= m; i++) {
      pmm *= -fact * somx2;  // Condon-Shortley phase
      fact += T(2);
    }
  }

  if (l == m) {
    return pmm;
  }

  // Seed: P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
  T pmm1 = x * T(2 * m + 1) * pmm;

  if (l == m + 1) {
    return pmm1;
  }

  // Recurrence: (l-m)*P_l^m = x*(2l-1)*P_{l-1}^m - (l+m-1)*P_{l-2}^m
  T pll = T(0);
  for (int ll = m + 2; ll <= l; ll++) {
    pll = (x * T(2 * ll - 1) * pmm1 - T(ll + m - 1) * pmm) / T(ll - m);
    pmm = pmm1;
    pmm1 = pll;
  }

  return pll;
}

// Compute normalization factor N_l^m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)
template <typename T>
T normalization(int l, int m) {
  // Compute (l-m)!/(l+m)! iteratively to avoid overflow
  T ratio = T(1);
  for (int i = l - m + 1; i <= l + m; i++) {
    ratio *= T(i);
  }
  return std::sqrt(T(2 * l + 1) / (T(4) * M_PI * ratio));
}

} // namespace detail

// Real template: computes real part of spherical harmonic
// For the real-type dispatch path, computes N * P_l^m(cos(theta)) * cos(m*phi)
// (or sin(|m|*phi) for negative m)
template <typename T>
T spherical_harmonic_y(T l, T m, T theta, T phi) {
  int l_int = static_cast<int>(l);
  int m_int = static_cast<int>(m);
  int abs_m = std::abs(m_int);

  if (l_int < 0 || abs_m > l_int) {
    return T(0);
  }

  T cos_theta = std::cos(theta);
  T plm = detail::associated_legendre_p(l_int, abs_m, cos_theta);
  T norm = detail::normalization<T>(l_int, abs_m);

  if (m_int > 0) {
    // Real spherical harmonic: sqrt(2) * N * P_l^m * cos(m*phi)
    return std::sqrt(T(2)) * norm * plm * std::cos(T(m_int) * phi);
  } else if (m_int == 0) {
    return norm * plm;
  } else {
    // Real spherical harmonic: sqrt(2) * N * P_l^|m| * sin(|m|*phi)
    return std::sqrt(T(2)) * norm * plm * std::sin(T(abs_m) * phi);
  }
}

// Complex template: computes full complex Y_l^m(theta, phi)
// Y_l^m = N_l^m * P_l^|m|(cos(theta)) * exp(i*m*phi)
// For negative m: Y_l^{-|m|} = (-1)^m * conj(Y_l^{|m|})
template <typename T>
c10::complex<T> spherical_harmonic_y(
    c10::complex<T> l, c10::complex<T> m,
    c10::complex<T> theta, c10::complex<T> phi) {
  int l_int = static_cast<int>(l.real());
  int m_int = static_cast<int>(m.real());
  int abs_m = std::abs(m_int);

  if (l_int < 0 || abs_m > l_int) {
    return c10::complex<T>(T(0), T(0));
  }

  T theta_real = theta.real();
  T phi_real = phi.real();

  T cos_theta = std::cos(theta_real);
  T plm = detail::associated_legendre_p(l_int, abs_m, cos_theta);
  T norm = detail::normalization<T>(l_int, abs_m);

  // Compute exp(i * |m| * phi)
  T angle = T(abs_m) * phi_real;
  c10::complex<T> exp_factor(std::cos(angle), std::sin(angle));

  c10::complex<T> result = c10::complex<T>(norm * plm, T(0)) * exp_factor;

  if (m_int < 0) {
    // Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
    T sign = (abs_m % 2 == 0) ? T(1) : T(-1);
    result = sign * std::conj(result);
  }

  return result;
}

} // namespace torchscience::kernel::special_functions
