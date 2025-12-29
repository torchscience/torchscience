# Design: Complex Support for torchscience.special_functions.gamma

**Status:** Approved
**Date:** 2025-12-29

## Overview

Add complex number support to the gamma function, enabling `torchscience.special_functions.gamma` to accept `complex64` and `complex128` tensors with full forward and backward (gradient) support.

## Use Cases

1. Scientific computing applications with inherently complex data
2. Complex differentiable losses (frequency-domain networks)
3. API completeness (matching scipy.special.gamma behavior)
4. Dependency for other functions requiring complex gamma

## Design Decisions

| Decision | Choice |
|----------|--------|
| Algorithm | Lanczos approximation (same as real) |
| File organization | Complex overloads in existing kernel files |
| Code structure | Overloaded functions (not `if constexpr`) |
| Pole handling | Tolerance-based detection (~1e-12 for double, ~1e-6 for float) |

## Files to Modify

| File | Change |
|------|--------|
| `kernel/special_functions/gamma.h` | Add complex overload using Lanczos |
| `kernel/special_functions/gamma_backward.h` | Add complex overload |
| `kernel/special_functions/gamma_backward_backward.h` | Add complex overload |
| `kernel/special_functions/digamma.h` | Add complex overload |
| `kernel/special_functions/trigamma.h` | Add complex overload |
| `cpu/special_functions/gamma.h` | Add `AT_DISPATCH_COMPLEX_TYPES` branches |

## Algorithm Details

### Complex Gamma (`gamma.h`)

```cpp
template <typename T>
c10::complex<T> gamma(c10::complex<T> z) {
  // 1. Pole detection: |imag(z)| < tol and real(z) is non-positive integer
  //    Return inf if pole detected

  // 2. Reflection formula for Re(z) < 0.5:
  //    Γ(z) = π / (sin(πz) * Γ(1-z))
  //    Need complex sin_pi helper for accuracy

  // 3. Lanczos for Re(z) >= 0.5:
  //    Same coefficients as real version (g=7, 9 terms)
  //    z_adj = z - 1
  //    x = c[0] + Σ c[i]/(z_adj + i)
  //    t = z_adj + g + 0.5
  //    Γ(z) = sqrt(2π) * t^(z_adj+0.5) * exp(-t) * x
}
```

**Helper needed:** `sin_pi(z)` for complex - computes `sin(π*z)` accurately using range reduction.

**Pole detection constants:**
- `kPoleToleranceFloat = 1e-6f`
- `kPoleToleranceDouble = 1e-12`

### Complex Digamma (`digamma.h`)

```cpp
template <typename T>
c10::complex<T> digamma(c10::complex<T> z) {
  // 1. Pole detection: same tolerance as gamma
  //    Return NaN at poles

  // 2. Reflection formula for Re(z) < 0.5:
  //    ψ(z) = ψ(1-z) - π*cot(πz)
  //    Need complex cot_pi helper

  // 3. Recurrence to shift Re(z) >= 6:
  //    ψ(z) = ψ(z+1) - 1/z

  // 4. Asymptotic expansion for Re(z) >= 6:
  //    ψ(z) ≈ ln(z) - 1/(2z) - Σ B_{2k}/(2k*z^{2k})
}
```

**Helper needed:** `cot_pi(z)` for complex.

### Complex Trigamma (`trigamma.h`)

```cpp
template <typename T>
c10::complex<T> trigamma(c10::complex<T> z) {
  // 1. Pole detection: same tolerance as gamma
  //    Return NaN at poles

  // 2. Reflection formula for Re(z) < 0.5:
  //    ψ₁(z) = -ψ₁(1-z) + π²/sin²(πz)

  // 3. Recurrence to shift Re(z) >= 6:
  //    ψ₁(z) = ψ₁(z+1) + 1/z²

  // 4. Asymptotic expansion for Re(z) >= 6:
  //    ψ₁(z) ≈ 1/z + 1/(2z²) + Σ B_{2k}/z^{2k+1}
}
```

### Backward Kernels

**`gamma_backward.h`:**
```cpp
template <typename T>
c10::complex<T> gamma_backward(c10::complex<T> g, c10::complex<T> z) {
  return g * gamma(z) * digamma(z);
}
```

**`gamma_backward_backward.h`:**
```cpp
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> gamma_backward_backward(
    c10::complex<T> gg, c10::complex<T> g, c10::complex<T> z) {
  auto gamma_z = gamma(z);
  auto psi = digamma(z);
  auto psi1 = trigamma(z);
  return {gg * gamma_z * psi, gg * g * gamma_z * (psi * psi + psi1)};
}
```

### CPU Dispatch (`cpu/special_functions/gamma.h`)

Add complex dispatch branch to each function:
```cpp
if (at::isComplexType(iterator.common_dtype())) {
  AT_DISPATCH_COMPLEX_TYPES(iterator.common_dtype(), "gamma_cpu_complex", [&] {
    at::native::cpu_kernel(iterator, kernel::special_functions::gamma<scalar_t>);
  });
} else {
  // existing AT_DISPATCH_FLOATING_TYPES_AND2 block
}
```

## Testing

**Existing tests that should pass:**
- `test_complex_conjugate_symmetry`
- `test_complex_positive_real_axis_matches_real`
- `test_complex_gradient_at_poles_nan`
- `test_complex_gradient_with_imaginary_part_finite`
- `test_complex_near_real_axis_at_poles`
- `test_complex_negative_half_integers`

**Tests to add/update:**
- Remove `skip`/`xfail` markers for complex tests
- Add `test_gradcheck_complex`
- Add `test_gradgradcheck_complex`

**Reference validation:**
- Compare against `scipy.special.gamma`
- Compare against `mpmath.gamma` for high-precision reference
