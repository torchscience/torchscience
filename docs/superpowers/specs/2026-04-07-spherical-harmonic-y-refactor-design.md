# Spherical Harmonic Y Refactor Design

## Goal

Refactor `spherical_harmonic_y` from a pure-Python implementation to a C++ kernel following the standard operator pattern used by all other operators in the codebase.

## Current State

`spherical_harmonic_y` is implemented entirely in Python (`src/torchscience/special_functions/_spherical_harmonic_y.py`). It exposes three functions:

- `spherical_harmonic_y(l, m, theta, phi, real=False)` — core computation
- `spherical_harmonic_y_cartesian(l, m, x, y, z, real=False)` — Cartesian coordinate wrapper
- `spherical_harmonic_y_all(l_max, theta, phi, real=False)` — batch computation up to degree l_max

The implementation calls `associated_legendre_polynomial_p` (also pure Python), has no C++ kernel, no dispatcher registration, no autograd/meta/autocast backends, and all tests are skipped.

## Design

### API

Single operator with signature:

```
spherical_harmonic_y(Tensor l, Tensor m, Tensor theta, Tensor phi) -> Tensor
```

- All four inputs are tensors. `l` and `m` are integer-valued but passed as tensors (following the established pattern from `polygamma` and `zernike_polynomial_z`).
- Output is always complex (matching SciPy's `sph_harm_y` convention).
- The Python wrapper promotes all inputs to complex dtype before calling `torch.ops.torchscience.spherical_harmonic_y`.
- Supports broadcasting across all four inputs.

### Removed

- `spherical_harmonic_y_cartesian` — users perform coordinate conversion themselves (`atan2`/`acos`).
- `spherical_harmonic_y_all` — users construct l/m tensors and rely on broadcasting.
- `real=False` parameter — always complex, like SciPy.
- Pure-Python Legendre computation — replaced by inline C++ recursion.

### C++ Kernel

**Forward** (`csrc/kernel/special_functions/spherical_harmonic_y.h`):

Computes Y_l^m(theta, phi) = N_l^m * P_l^m(cos theta) * exp(i*m*phi) where:

- N_l^m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) is the normalization factor
- P_l^m is the associated Legendre polynomial (Condon-Shortley phase included)
- exp(i*m*phi) is the azimuthal factor

Associated Legendre polynomial computed via inline recursion:

1. Seed: P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
2. Seed: P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
3. Recurrence: (l-m)*P_l^m(x) = x*(2l-1)*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)

Negative m handled via: Y_l^{-|m|} = (-1)^m * conj(Y_l^{|m|})

Template structure (following `zernike_polynomial_z`):

- Real template: `T spherical_harmonic_y(T l, T m, T theta, T phi)` — computes real spherical harmonic (N * P_l^m(cos theta) * cos(m*phi) or sin(|m|*phi))
- Complex template: `c10::complex<T> spherical_harmonic_y(c10::complex<T> l, ...)` — computes full complex Y_l^m with exp(i*m*phi)

Both cast l/m to int via `static_cast<int>`, same as `polygamma`.

### Gradient Kernels

**First-order backward** (`spherical_harmonic_y_backward.h`):

Returns four gradient tensors (matching quaternary macro pattern):

- grad_l = 0 (discrete parameter)
- grad_m = 0 (discrete parameter)
- grad_phi = grad_output * i*m * Y_l^m(theta, phi)
- grad_theta = grad_output * N_l^m * dP_l^m(cos theta)/d(theta) * exp(i*m*phi)

The Legendre derivative uses the recurrence:

dP_l^m(cos theta)/d(theta) = [l*cos(theta)*P_l^m(cos theta) - (l+m)*P_{l-1}^m(cos theta)] / sin(theta)

The backward kernel recomputes P_l^m and P_{l-1}^m via the same recursion as the forward kernel.

**Second-order backward** (`spherical_harmonic_y_backward_backward.h`):

- d^2Y/d(phi)^2 = -m^2 * Y_l^m
- d^2Y/d(theta)^2 via second derivatives of Legendre polynomials using the same recurrence
- d^2Y/d(theta)*d(phi) = i*m * dY/d(theta) (cross term)
- All partials involving l or m remain zero

### Backend Registration

One macro invocation per backend in the corresponding `special_functions.h`:

| Backend | File | Macro |
|---------|------|-------|
| CPU | `csrc/cpu/special_functions.h` | `TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX` |
| Autograd | `csrc/autograd/special_functions.h` | `TORCHSCIENCE_AUTOGRAD_POINTWISE_QUATERNARY_OPERATOR` |
| Meta | `csrc/meta/special_functions.h` | `TORCHSCIENCE_META_POINTWISE_QUATERNARY_OPERATOR` |
| Autocast | `csrc/autocast/special_functions.h` | `TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR` |

Schema definitions in `csrc/special_functions.cpp`:

```cpp
m.def("spherical_harmonic_y(Tensor l, Tensor m, Tensor theta, Tensor phi) -> Tensor");
m.def("spherical_harmonic_y_backward(Tensor grad, Tensor l, Tensor m, Tensor theta, Tensor phi) -> (Tensor, Tensor, Tensor, Tensor)");
m.def("spherical_harmonic_y_backward_backward(Tensor grad_0, Tensor grad_1, Tensor grad_2, Tensor grad_3, Tensor l, Tensor m, Tensor theta, Tensor phi) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
```

### Testing

Test file: `tests/torchscience/special_functions/test__spherical_harmonic_y.py` (rewritten).

- **Forward correctness**: compare against `scipy.special.sph_harm_y(n, m, theta, phi)` for l=0..6, m=-l..l, various theta in [0, pi] and phi in [0, 2*pi]
- **Gradient correctness**: `torch.autograd.gradcheck` on theta and phi; `torch.autograd.gradgradcheck` for second-order
- **Special values**: theta=0 and theta=pi (poles), Y_0^0 = 1/sqrt(4*pi), m=0 cases, negative m
- **Broadcasting**: different-shaped l, m, theta, phi tensors
- **Meta tensors**: shape inference
- **Autocast**: mixed precision behavior

All currently skipped tests are removed — the refactored operator should pass.

### Files Summary

**Create:**
- `csrc/kernel/special_functions/spherical_harmonic_y.h`
- `csrc/kernel/special_functions/spherical_harmonic_y_backward.h`
- `csrc/kernel/special_functions/spherical_harmonic_y_backward_backward.h`

**Modify:**
- `csrc/special_functions.cpp` — schema definitions
- `csrc/cpu/special_functions.h` — CPU macro
- `csrc/autograd/special_functions.h` — autograd macro
- `csrc/meta/special_functions.h` — meta macro
- `csrc/autocast/special_functions.h` — autocast macro
- `src/torchscience/special_functions/_spherical_harmonic_y.py` — rewrite to thin wrapper
- `src/torchscience/special_functions/__init__.py` — remove _cartesian and _all exports
- `tests/torchscience/special_functions/test__spherical_harmonic_y.py` — rewrite tests
