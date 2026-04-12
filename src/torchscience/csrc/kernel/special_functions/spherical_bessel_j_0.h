#pragma once

#include <cmath>
#include "cmath_compat.h"
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
C10_HOST_DEVICE constexpr T spherical_bessel_j_0_eps();

template <>
C10_HOST_DEVICE constexpr float spherical_bessel_j_0_eps<float>() { return 1e-6f; }

template <>
C10_HOST_DEVICE constexpr double spherical_bessel_j_0_eps<double>() { return 1e-12; }

template <>
C10_HOST_DEVICE inline c10::Half spherical_bessel_j_0_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
C10_HOST_DEVICE inline c10::BFloat16 spherical_bessel_j_0_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Spherical Bessel function of the first kind, order 0
// j_0(z) = sin(z) / z
template <typename T>
C10_HOST_DEVICE T spherical_bessel_j_0(T z) {
    if (cmath_compat::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::spherical_bessel_j_0_eps<T>();

    // For small z, use Taylor series: j_0(z) = 1 - z^2/6 + z^4/120 - ...
    if (std::abs(z) < eps) {
        T z2 = z * z;
        return T(1) - z2 / T(6) + z2 * z2 / T(120);
    }

    return std::sin(z) / z;
}

// Complex version
template <typename T>
C10_HOST_DEVICE c10::complex<T> spherical_bessel_j_0(c10::complex<T> z) {
    const T eps = detail::spherical_bessel_j_0_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        return c10::complex<T>(T(1), T(0)) - z2 / c10::complex<T>(T(6), T(0))
               + z2 * z2 / c10::complex<T>(T(120), T(0));
    }

    return std::sin(z) / z;
}

} // namespace torchscience::kernel::special_functions
