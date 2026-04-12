#pragma once

// MSVC compatibility wrappers for <cmath> classification functions.
//
// MSVC's <cmath> provides overloads of isfinite/isinf/isnan for integral
// types (wchar_t, char16_t, char32_t, etc.), which causes "ambiguous call
// to overloaded function" errors (C2668) when these functions are called
// with template parameter types. Wrapping through explicitly-typed inline
// functions resolves the ambiguity on all compilers.

#include <cmath>
#include <c10/macros/Macros.h>

namespace torchscience::kernel::special_functions {
namespace cmath_compat {

template <typename T>
C10_HOST_DEVICE inline bool isfinite(T x) {
  return std::isfinite(static_cast<double>(x));
}

template <>
C10_HOST_DEVICE inline bool isfinite<float>(float x) {
  return std::isfinite(x);
}

template <>
C10_HOST_DEVICE inline bool isfinite<double>(double x) {
  return std::isfinite(x);
}

template <typename T>
C10_HOST_DEVICE inline bool isinf(T x) {
  return std::isinf(static_cast<double>(x));
}

template <>
C10_HOST_DEVICE inline bool isinf<float>(float x) {
  return std::isinf(x);
}

template <>
C10_HOST_DEVICE inline bool isinf<double>(double x) {
  return std::isinf(x);
}

template <typename T>
C10_HOST_DEVICE inline bool isnan(T x) {
  return std::isnan(static_cast<double>(x));
}

template <>
C10_HOST_DEVICE inline bool isnan<float>(float x) {
  return std::isnan(x);
}

template <>
C10_HOST_DEVICE inline bool isnan<double>(double x) {
  return std::isnan(x);
}

} // namespace cmath_compat
} // namespace torchscience::kernel::special_functions
