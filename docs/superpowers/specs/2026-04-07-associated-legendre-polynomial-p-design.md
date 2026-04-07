# Associated Legendre Polynomial P Refactor Design

## Goal

Refactor `associated_legendre_polynomial_p` from a pure-Python implementation to a C++ kernel following the standard operator pattern used by all other operators in the codebase.

## Current State

`associated_legendre_polynomial_p` is implemented entirely in Python (`src/torchscience/special_functions/_associated_legendre_polynomial_p.py`). It exposes two functions:

- `associated_legendre_polynomial_p(n, m, x, normalized=False)` — core computation with scalar int n, m
- `associated_legendre_polynomial_p_all(n_max, x, normalized=False)` — batch computation up to degree n_max

The implementation uses three-term recurrence, has no C++ kernel, no dispatcher registration, no autograd/meta/autocast backends.

## Design

### API

Single operator with signature:

```
associated_legendre_polynomial_p(Tensor n, Tensor m, Tensor x) -> Tensor
```

- All three inputs are tensors. `n` and `m` are integer-valued but passed as tensors (following the established pattern from `polygamma`, `zernike_polynomial_z`, and `spherical_harmonic_y`).
- Supports broadcasting across all three inputs.
- Condon-Shortley phase convention (includes the (-1)^m factor).
- Negative m handled internally via symmetry: P_n^{-|m|}(x) = (-1)^m * (n-m)!/(n+m)! * P_n^{|m|}(x).

### Removed

- `associated_legendre_polynomial_p_all` — users construct n/m tensors and rely on broadcasting.
- `normalized` parameter — users apply `sqrt((2n+1)*(n-m)! / (2*(n+m)!))` themselves.

### C++ Kernel

**Forward** (`csrc/kernel/special_functions/associated_legendre_polynomial_p.h`):

Computes P_n^m(x) via three-term recursion with Condon-Shortley phase:

1. Cast n, m to int via `static_cast<int>`.
2. Handle negative m: compute for |m|, then apply symmetry relation P_n^{-|m|}(x) = (-1)^m * (n-m)!/(n+m)! * P_n^{|m|}(x).
3. Seed: P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2).
4. Seed: P_{m+1}^m(x) = x * (2m+1) * P_m^m(x).
5. Recurrence: (k-m+1) * P_{k+1}^m(x) = (2k+1) * x * P_k^m(x) - (k+m) * P_{k-1}^m(x).

Template structure (following `zernike_polynomial_z`):

- Real template: `T associated_legendre_polynomial_p(T n, T m, T x)` — standard computation.
- Complex template: `c10::complex<T> associated_legendre_polynomial_p(c10::complex<T> n, c10::complex<T> m, c10::complex<T> x)` — extracts real parts for n, m, computes with real arithmetic, returns real result cast to complex.

### Gradient Kernels

**First-order backward** (`associated_legendre_polynomial_p_backward.h`):

Returns three gradient tensors:

- grad_n = 0 (discrete parameter)
- grad_m = 0 (discrete parameter)
- grad_x = grad_output * [n*x*P_n^m(x) - (n+m)*P_{n-1}^m(x)] / (x^2 - 1)

The backward kernel recomputes P_n^m and P_{n-1}^m via the same recursion as the forward kernel.

Special case at x = +/-1: the denominator x^2 - 1 = 0 causes division by zero. Handle by clamping x to avoid the singularity (e.g., clamp to +/-(1 - eps)), matching the approach used for pole handling elsewhere in the codebase.

**Second-order backward** (`associated_legendre_polynomial_p_backward_backward.h`):

- d^2P/dx^2 derived from differentiating the first-order gradient formula.
- All partials involving n or m remain zero.
- Cross terms d^2P/(dn*dx) and d^2P/(dm*dx) are zero.

### Backend Registration

One macro invocation per backend in the corresponding `special_functions.h`:

| Backend | File | Macro |
|---------|------|-------|
| CPU | `csrc/cpu/special_functions.h` | `TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX` |
| Autograd | `csrc/autograd/special_functions.h` | `TORCHSCIENCE_AUTOGRAD_POINTWISE_TERNARY_OPERATOR` |
| Meta | `csrc/meta/special_functions.h` | `TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR` |
| Autocast | `csrc/autocast/special_functions.h` | `TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR` |

Schema definitions in `csrc/special_functions.cpp`:

```cpp
m.def("associated_legendre_polynomial_p(Tensor n, Tensor m, Tensor x) -> Tensor");
m.def("associated_legendre_polynomial_p_backward(Tensor grad_output, Tensor n, Tensor m, Tensor x) -> (Tensor, Tensor, Tensor)");
m.def("associated_legendre_polynomial_p_backward_backward(Tensor gg_n, Tensor gg_m, Tensor gg_x, Tensor grad_output, Tensor n, Tensor m, Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
```

### Testing

Test file: `tests/torchscience/special_functions/test__associated_legendre_polynomial_p.py` (rewritten).

- **Forward correctness**: compare against `scipy.special.lpmv(m, n, x)` for n=0..6, m=0..n, various x in [-1, 1]
- **Negative m**: verify symmetry relation P_n^{-m} = (-1)^m * (n-m)!/(n+m)! * P_n^m
- **Special values**: x=+/-1 (endpoints), P_0^0=1, P_1^0=x, P_1^1=-sqrt(1-x^2), P_2^0=(3x^2-1)/2
- **Gradient correctness**: `torch.autograd.gradcheck` on x; verify grad_n=0 and grad_m=0
- **Second-order gradients**: `torch.autograd.gradgradcheck` on x
- **Broadcasting**: different-shaped n, m, x tensors
- **Meta tensors**: shape inference
- **Autocast**: mixed precision
- Uses the `OpTestCase` / `OperatorDescriptor` framework

### Files Summary

**Create:**
- `csrc/kernel/special_functions/associated_legendre_polynomial_p.h`
- `csrc/kernel/special_functions/associated_legendre_polynomial_p_backward.h`
- `csrc/kernel/special_functions/associated_legendre_polynomial_p_backward_backward.h`

**Modify:**
- `csrc/special_functions.cpp` — schema definitions
- `csrc/cpu/special_functions.h` — CPU macro + includes
- `csrc/autograd/special_functions.h` — autograd macro
- `csrc/meta/special_functions.h` — meta macro
- `csrc/autocast/special_functions.h` — autocast macro
- `src/torchscience/special_functions/_associated_legendre_polynomial_p.py` — rewrite to thin wrapper, remove `_all`
- `src/torchscience/special_functions/__init__.py` — remove `_all` export
- `tests/torchscience/special_functions/test__associated_legendre_polynomial_p.py` — rewrite tests
