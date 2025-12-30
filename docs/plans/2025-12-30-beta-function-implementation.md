# Beta Function Implementation

**Status:** Approved
**Date:** 2025-12-30

## Overview

Implement `torchscience.special_functions.beta` - the Euler beta function with full autograd support.

## Mathematical Definition

**Forward:**
```
B(a, b) = Γ(a)Γ(b) / Γ(a+b) = exp(log_beta(a, b))
```

**Backward (first-order gradients):**
```
∂B/∂a = B(a,b) · [ψ(a) - ψ(a+b)]
∂B/∂b = B(a,b) · [ψ(b) - ψ(a+b)]
```
where ψ is the digamma function.

**Backward-backward (second-order gradients):**
```
∂²B/∂a² = B(a,b) · [(ψ(a) - ψ(a+b))² + ψ'(a) - ψ'(a+b)]
∂²B/∂b² = B(a,b) · [(ψ(b) - ψ(a+b))² + ψ'(b) - ψ'(a+b)]
∂²B/∂a∂b = B(a,b) · [(ψ(a) - ψ(a+b))(ψ(b) - ψ(a+b)) - ψ'(a+b)]
```
where ψ' is the trigamma function.

**Dtype support:** float16, bfloat16, float32, float64, complex64, complex128

## Implementation Structure

### Files to Create

1. `src/torchscience/csrc/kernel/special_functions/beta_backward.h` - gradient kernel
2. `src/torchscience/csrc/kernel/special_functions/beta_backward_backward.h` - Hessian kernel
3. `src/torchscience/special_functions/_beta.py` - Python API with docstring
4. `tests/torchscience/special_functions/test__beta.py` - test suite

### Files to Modify

1. `src/torchscience/csrc/torchscience.cpp` - add schema definitions:
   - `beta(Tensor a, Tensor b) -> Tensor`
   - `beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)`
   - `beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)`

2. `src/torchscience/csrc/cpu/special_functions.h`:
   - Add kernel includes for beta, beta_backward, beta_backward_backward
   - Add `TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(beta, a, b)`

3. `src/torchscience/csrc/meta/special_functions.h`:
   - Add `TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(beta, a, b)`

4. `src/torchscience/csrc/autograd/special_functions.h`:
   - Add `TORCHSCIENCE_AUTOGRAD_POINTWISE_BINARY_OPERATOR(beta, Beta, a, b)`

5. `src/torchscience/csrc/autocast/special_functions.h`:
   - Add `TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(beta, a, b)`

6. `src/torchscience/special_functions/__init__.py`:
   - Import and export `beta`

7. Sparse/quantized backends (8 files):
   - `sparse/coo/cpu/special_functions.h`
   - `sparse/coo/cuda/special_functions.h`
   - `sparse/csr/cpu/special_functions.h`
   - `sparse/csr/cuda/special_functions.h`
   - `quantized/cpu/special_functions.h`
   - `quantized/cuda/special_functions.h`

## Testing Strategy

### Test Cases

1. **Correctness** - compare against `torch.exp(torch.special.betaln(a, b))` for real inputs
2. **Special values** - B(1,1)=1, B(1,n)=1/n, B(0.5,0.5)=π
3. **Symmetry** - B(a,b) = B(b,a)
4. **Complex inputs** - verify against scipy.special.beta where available
5. **Gradient check** - `torch.autograd.gradcheck` for first-order
6. **Gradgrad check** - `torch.autograd.gradgradcheck` for second-order
7. **Dtype coverage** - float32, float64, complex64, complex128
8. **Edge cases** - large values (overflow to inf), values near poles

### Known Limitations

- Poles at a=0,-1,-2,... and b=0,-1,-2,... (returns inf)
- Overflow for large positive a,b where Γ(a)Γ(b) overflows before division

## Existing Infrastructure

- `kernel/special_functions/beta.h` - forward kernel already exists
- `kernel/special_functions/log_beta.h` - used by beta.h
- `kernel/special_functions/digamma.h` - needed for backward
- `kernel/special_functions/trigamma.h` - needed for backward_backward
