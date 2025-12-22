#pragma once

/*
 * Type Traits for Unified Real/Complex Handling
 *
 * DESIGN NOTES:
 *
 * This header provides template utilities that enable writing generic code
 * that works uniformly for both real (float, double) and complex
 * (c10::complex<float>, c10::complex<double>) types.
 *
 * Key utilities:
 * - make_scalar_for: Creates a scalar value of the same type as a reference
 * - get_real: Extracts real part (identity for real types)
 * - is_near_zero: Checks if a value is approximately zero
 *
 * These utilities reduce code duplication between real and complex
 * implementations of special functions.
 */

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace torchscience::impl::special_functions {

// ============================================================================
// make_scalar_for: Create a scalar value matching the type of a reference
// ============================================================================

/**
 * Creates a scalar value of type T from a real value.
 * For complex types, creates a complex number with zero imaginary part.
 * For real types, simply returns the value cast to T.
 *
 * Usage:
 *   T x = ...;
 *   T one = make_scalar_for(x, 1.0);  // Creates 1.0 or (1.0, 0.0)
 *
 * @param ref Reference value (used only for type deduction)
 * @param val The real value to convert
 * @return A scalar of type T with value val (and zero imaginary part if complex)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<c10::is_complex<T>::value, T>
make_scalar_for(T /*ref*/, typename T::value_type val) {
  return T(val, typename T::value_type(0));
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
make_scalar_for(T /*ref*/, typename c10::scalar_value_type<T>::type val) {
  return T(val);
}

// ============================================================================
// get_real: Extract real part (identity for real types)
// ============================================================================

/**
 * Extracts the real part of a value.
 * For real types, returns the value unchanged.
 * For complex types, returns the real component.
 *
 * @param x Input value
 * @return Real part of x
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
get_real(T x) {
  return x;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T get_real(c10::complex<T> x) {
  return x.real();
}

// ============================================================================
// get_imag: Extract imaginary part (zero for real types)
// ============================================================================

/**
 * Extracts the imaginary part of a value.
 * For real types, returns zero.
 * For complex types, returns the imaginary component.
 *
 * @param x Input value
 * @return Imaginary part of x (0 for real types)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
get_imag(T /*x*/) {
  return T(0);
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T get_imag(c10::complex<T> x) {
  return x.imag();
}

// ============================================================================
// scalar_value_t: Alias for the underlying real type
// ============================================================================

/**
 * Type alias for the underlying real type.
 * For real types T, this is T.
 * For complex<T>, this is T.
 */
template <typename T>
using scalar_value_t = typename c10::scalar_value_type<T>::type;

// ============================================================================
// make_complex: Create complex from real/imag parts
// ============================================================================

/**
 * Creates a complex number from real and imaginary parts.
 * For real types, returns only the real part (ignores imaginary).
 * For complex types, constructs the complex number.
 *
 * @param real_part Real component
 * @param imag_part Imaginary component (ignored for real types)
 * @return Value of type T
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
make_complex(T real_part, T /*imag_part*/) {
  return real_part;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> make_complex(T real_part, T imag_part) {
  return c10::complex<T>(real_part, imag_part);
}

// ============================================================================
// conj_if_complex: Conjugate for complex, identity for real
// ============================================================================

/**
 * Returns the complex conjugate for complex types, identity for real types.
 *
 * @param x Input value
 * @return Conjugate of x (unchanged for real types)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
conj_if_complex(T x) {
  return x;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
c10::complex<T> conj_if_complex(c10::complex<T> x) {
  return std::conj(x);
}

// ============================================================================
// abs_squared: |x|^2 for both real and complex
// ============================================================================

/**
 * Computes the squared absolute value.
 * For real types: x^2
 * For complex types: |z|^2 = Re(z)^2 + Im(z)^2
 *
 * @param x Input value
 * @return Squared absolute value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::enable_if_t<!c10::is_complex<T>::value, T>
abs_squared(T x) {
  return x * x;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T abs_squared(c10::complex<T> x) {
  return x.real() * x.real() + x.imag() * x.imag();
}

}  // namespace torchscience::impl::special_functions
