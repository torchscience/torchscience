# Hypergeometric 2F1 Implementation Design

**Date:** 2025-12-29
**Status:** Approved
**Scope:** Full implementation of the Gaussian hypergeometric function with autograd support

## Overview

Implement the Gaussian hypergeometric function ₂F₁(a,b;c;z) with:
- Full complex plane support for z
- Complex parameters a, b, c
- Adaptive accuracy matching input dtype
- Tiered approach: fast paths for common cases, robust fallbacks for difficult regions
- Forward, backward, and backward_backward for autograd

## Algorithm: Series + Linear Transformations

Use the power series for |z| < 0.5, then apply Kummer linear transformations to map other z regions back to the convergent disk.

### High-Level Architecture

Three layers:

1. **Entry point functions** (`hypergeometric_2_f_1_forward`, `_backward`, `_backward_backward`)
   - Validate inputs, handle dtype dispatch
   - Use PyTorch's `TensorIterator` for vectorized element-wise computation
   - Match the pattern in `gamma.h`

2. **Core scalar kernel** (`hyp2f1_kernel<T>(a, b, c, z)`)
   - Single complex/real evaluation
   - Dispatches to the appropriate algorithm based on parameter region

3. **Region-specific algorithms**
   - `hyp2f1_series()` - Direct power series for |z| < 0.5
   - `hyp2f1_transform()` - Apply transformations, recurse
   - `hyp2f1_special()` - Handle integer a, b, c cases
   - `hyp2f1_asymptotic()` - Large parameter expansions (optional)

### Data Flow

```
forward(a,b,c,z)
  → TensorIterator
    → hyp2f1_kernel(a,b,c,z)
      → classify_region(z)
      → if |z| < 0.5: hyp2f1_series(a,b,c,z)
      → else: select_transform(a,b,c,z) → hyp2f1_kernel(a',b',c',z')
```

## Series Computation

The power series:

```
₂F₁(a,b;c;z) = Σ_{n=0}^∞ (a)_n (b)_n / (c)_n · zⁿ / n!
```

Implementation:

```cpp
template <typename T>
T hyp2f1_series(T a, T b, T c, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    term *= (a + T(n)) * (b + T(n)) / ((c + T(n)) * T(n + 1)) * z;
    sum += term;

    if (std::abs(term) < epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}
```

Key details:
- **Convergence region**: |z| < 1, but use |z| < 0.5 for stability
- **Termination**: Series terminates if a or b is non-positive integer
- **Pole handling**: Returns Inf if c is non-positive integer
- **Precision**: epsilon<T>() is ~1e-7 for float, ~1e-15 for double

## Transformation Selection

When |z| >= 0.5, apply linear transformations:

| Region | Transformation | Maps z to |
|--------|----------------|-----------|
| z near 1 | `z → 1-z` (Euler) | near 0 |
| z > 1 | `z → 1/z` (Pfaff) | \|z\| < 1 |
| z < 0, large | `z → z/(z-1)` | (0, 1) |
| z near 1, from above | `z → 1-1/z` | near 0 |

Selection logic:

```cpp
template <typename T>
T hyp2f1_kernel(T a, T b, T c, T z) {
  if (std::abs(z) < T(0.5)) {
    return hyp2f1_series(a, b, c, z);
  }

  if (std::abs(T(1) - z) < T(0.5)) {
    return hyp2f1_near_one(a, b, c, z);
  }

  if (std::abs(z) > T(1)) {
    return hyp2f1_large_z(a, b, c, z);
  }

  return hyp2f1_pfaff(a, b, c, z);
}
```

## Special Case Handling

### Terminating Series (a or b = -m)

```cpp
template <typename T>
bool is_nonpositive_integer(T x) {
  return std::real(x) <= 0 &&
         std::abs(x - std::round(std::real(x))) < epsilon<T>();
}
```

When a = -m, compute exactly m+1 terms.

### Poles (c = -n)

Return infinity unless the pole is cancelled by a or b being a "smaller" non-positive integer.

### Special Values

- `₂F₁(a,b;c;0) = 1`
- `₂F₁(a,b;b;z) = (1-z)^(-a)`
- `₂F₁(a,b;a;z) = (1-z)^(-b)`

## Gradient Computation

### Derivative with respect to z

```
∂/∂z ₂F₁(a,b;c;z) = (ab/c) · ₂F₁(a+1, b+1; c+1; z)
```

### Derivatives with respect to a, b, c

Compute alongside forward pass using running digamma sums:

```cpp
template <typename T>
std::tuple<T, T, T, T, T> hyp2f1_with_grads(T a, T b, T c, T z) {
  T sum = T(1), da = T(0), db = T(0), dc = T(0);
  T term = T(1);
  T psi_a = T(0), psi_b = T(0), psi_c = T(0);

  for (int n = 0; n < max_iter; ++n) {
    if (n > 0) {
      psi_a += T(1) / (a + T(n - 1));
      psi_b += T(1) / (b + T(n - 1));
      psi_c += T(1) / (c + T(n - 1));
    }

    da += term * psi_a;
    db += term * psi_b;
    dc -= term * psi_c;

    term *= (a + T(n)) * (b + T(n)) / ((c + T(n)) * T(n + 1)) * z;
    sum += term;

    if (converged(term, sum)) break;
  }

  T dz = (a * b / c) * hyp2f1_kernel(a + 1, b + 1, c + 1, z);

  return {sum, da, db, dc, dz};
}
```

## Second-Order Gradients

For backward_backward:

- `∂(grad_z)/∂z` uses `₂F₁(a+2,b+2;c+2;z)`
- Second-order parameter derivatives are expensive (involve higher-order hypergeometric functions)
- Initial implementation may return zeros or raise errors for pure parameter second derivatives

## Testing Strategy

Uses `torchscience.testing` infrastructure with `OpTestCase`:

**Automatic tests from mixins:**
- `AutogradMixin`: gradcheck, gradgradcheck
- `DtypeMixin`: float32, float64, complex64, complex128
- `DeviceMixin`: CPU, CUDA
- `VmapMixin`: batched computation
- `TorchCompileMixin`: torch.compile compatibility
- `SymPyReferenceMixin`: symbolic verification
- `IdentityMixin`: functional identities

**Custom tests:**
- SciPy comparison across z regions
- Transformation boundary tests
- Terminating series verification
- Complex parameter tests (mpmath reference)

**Test configuration in OperatorDescriptor:**
```python
tolerances=ToleranceConfig(
    float64_rtol=1e-10,
    gradcheck_rtol=1e-4,
)
skip_tests={
    "test_sparse_coo_basic",  # 2F1 has special behavior at zeros
    "test_quantized_basic",   # Precision loss too high
}
```

## Error Handling

| Case | Detection | Handling |
|------|-----------|----------|
| c = 0, -1, -2, ... | `is_nonpositive_integer(c)` | Return ±∞ (unless cancelled) |
| a or b = 0, -1, -2, ... | `is_nonpositive_integer(a)` | Terminating series (exact) |
| z = 1 exactly | `z == T(1)` | Gauss formula if Re(c-a-b) > 0 |
| z very close to 1 | `abs(1-z) < 1e-10` | Switch to 1-z transformation |
| Large parameters | `max(abs(a),abs(b),abs(c)) > 100` | Asymptotic or warn |
| Overflow in term | `!std::isfinite(term)` | Return current sum or ∞ |
| Convergence failure | `n >= max_iter` | Return best estimate, warn |

**Branch cut:** For real z > 1, approach from above (consistent with SciPy).

## File Structure

```
src/torchscience/csrc/cpu/special_functions/
  hypergeometric_2_f_1.h  # CPU kernel implementation

src/torchscience/csrc/meta/special_functions/
  hypergeometric_2_f_1.h  # Shape inference

src/torchscience/csrc/autograd/special_functions/
  hypergeometric_2_f_1.h  # Autograd wrapper

src/torchscience/csrc/autocast/special_functions/
  hypergeometric_2_f_1.h  # Autocast dispatch

tests/torchscience/special_functions/
  test__hypergeometric_2_f_1.py  # Test suite (already exists)
```

## References

- DLMF Chapter 15: https://dlmf.nist.gov/15
- SciPy hyp2f1 implementation
- mpmath hypergeometric implementation
- "Computation of Hypergeometric Functions" by John Pearson (thesis)
