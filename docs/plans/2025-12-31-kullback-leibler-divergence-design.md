# Design: torchscience.information_theory.kullback_leibler_divergence

**Status:** Complete

## Overview

Add Kullback-Leibler divergence and Jensen-Shannon divergence to the `torchscience.information_theory` module with full autograd support, flexible input representations, and pairwise computation mode.

## API Surface

### kullback_leibler_divergence

```python
def kullback_leibler_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal["probability", "log_probability", "logits"] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
) -> Tensor:
```

**Parameters:**

- `p`: First probability distribution (or batch of distributions)
- `q`: Second probability distribution (or batch of distributions)
- `dim`: Dimension along which the probability distribution is defined (default: -1)
- `input_type`: How to interpret input tensors:
  - `"probability"`: Direct probability mass functions (will be epsilon-clamped)
  - `"log_probability"`: Log-probabilities (will be exponentiated)
  - `"logits"`: Unnormalized logits (softmax applied)
- `reduction`: How to reduce the output:
  - `"none"`: Return per-sample divergences
  - `"mean"`: Mean over all elements
  - `"batchmean"`: Mean over batch dimension (mathematically correct KL)
  - `"sum"`: Sum over all elements
- `pairwise`: If True, compute all-pairs divergence matrix

### jensen_shannon_divergence

```python
def jensen_shannon_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal["probability", "log_probability", "logits"] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
    base: Optional[float] = None,
) -> Tensor:
```

Same parameters as KL divergence, plus:

- `base`: Logarithm base for output scaling. `None` for natural log (output in nats), `2` for bits. JS divergence is bounded by `log(2)` in the specified base.

## Mathematical Definition

### Kullback-Leibler Divergence

$$D_{KL}(P \| Q) = \sum_{i} p_i \log\left(\frac{p_i}{q_i}\right) = \sum_{i} p_i \log(p_i) - \sum_{i} p_i \log(q_i)$$

### Jensen-Shannon Divergence

$$D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$ is the mixture distribution.

Properties:
- Symmetric: $D_{JS}(P \| Q) = D_{JS}(Q \| P)$
- Bounded: $0 \leq D_{JS} \leq \log(2)$
- Square root is a proper metric

## Numerical Stability

### Epsilon Clamping

Dtype-dependent epsilon values applied after preprocessing:

| dtype | epsilon |
|-------|---------|
| float16 | 1e-4 |
| bfloat16 | 1e-3 |
| float32 | 1e-7 |
| float64 | 1e-15 |

Clamping: `p_clamped = clamp(p, min=eps)`, `q_clamped = clamp(q, min=eps)`

### Input Type Preprocessing

1. `"probability"`: Apply epsilon clamping directly
2. `"log_probability"`: `p = exp(log_p)`, then clamp
3. `"logits"`: `p = softmax(logits, dim=dim)`, then clamp

## Shape Behavior

### Standard Mode (pairwise=False)

- Input: `p: (*, n)`, `q: (*, n)` (broadcastable)
- Output with `reduction="none"`: `(*,)`
- Output with other reductions: scalar

### Pairwise Mode (pairwise=True)

- Input: `p: (m, n)`, `q: (k, n)`
- Output with `reduction="none"`: `(m, k)`
- `output[i, j] = KL(p[i] || q[j])`

## Gradient Formulas

### KL Divergence First-Order

$$\frac{\partial D_{KL}}{\partial p_i} = \log(p_i) - \log(q_i) + 1$$

$$\frac{\partial D_{KL}}{\partial q_i} = -\frac{p_i}{q_i}$$

### KL Divergence Second-Order

$$\frac{\partial^2 D_{KL}}{\partial p_i^2} = \frac{1}{p_i}$$

$$\frac{\partial^2 D_{KL}}{\partial q_i^2} = \frac{p_i}{q_i^2}$$

$$\frac{\partial^2 D_{KL}}{\partial p_i \partial q_i} = -\frac{1}{q_i}$$

## File Structure

### Python API

```
src/torchscience/information_theory/
├── __init__.py
├── _kullback_leibler_divergence.py
└── _jensen_shannon_divergence.py
```

### C++ Backend

```
src/torchscience/csrc/
├── kernel/information_theory/
│   ├── kullback_leibler_divergence.h
│   ├── kullback_leibler_divergence_backward.h
│   ├── kullback_leibler_divergence_backward_backward.h
│   ├── jensen_shannon_divergence.h
│   ├── jensen_shannon_divergence_backward.h
│   └── jensen_shannon_divergence_backward_backward.h
├── cpu/information_theory/
│   ├── kullback_leibler_divergence.h
│   └── jensen_shannon_divergence.h
├── meta/information_theory/
│   ├── kullback_leibler_divergence.h
│   └── jensen_shannon_divergence.h
└── autograd/information_theory/
    ├── kullback_leibler_divergence.h
    └── jensen_shannon_divergence.h
```

### Operator Registration

```cpp
// In torchscience.cpp
m.def("kullback_leibler_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
m.def("jensen_shannon_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise, float? base) -> Tensor");
```

## Testing Strategy

### Test File

`tests/torchscience/information_theory/test__kullback_leibler_divergence.py`

### Test Categories

1. **Correctness tests:**
   - Compare against `torch.nn.functional.kl_div` for compatible cases
   - Compare against `scipy.stats.entropy` for discrete distributions
   - Known analytical cases (uniform vs uniform, Bernoulli distributions)

2. **Input type tests:**
   - Verify `input_type="probability"` uses values directly
   - Verify `input_type="log_probability"` exponentiates correctly
   - Verify `input_type="logits"` applies softmax correctly

3. **Reduction tests:**
   - `reduction="none"`: verify output shape matches batch dimensions
   - `reduction="sum"`: verify scalar output with correct sum
   - `reduction="mean"`: verify mean over all elements
   - `reduction="batchmean"`: verify mean over batch dimension

4. **Pairwise tests:**
   - Verify output shape `(m, k)` for inputs `(m, n)` and `(k, n)`
   - Verify `output[i, j] == kl(p[i], q[j])` elementwise

5. **Gradient tests:**
   - `torch.autograd.gradcheck` for first-order gradients
   - `torch.autograd.gradgradcheck` for second-order gradients
   - Test gradient flow through all input types

6. **Edge cases:**
   - Near-zero probabilities (verify clamping prevents NaN/inf)
   - Single-element distributions
   - Various dtypes: float32, float64, float16, bfloat16

## Validation (Python Layer)

- Check `p` and `q` are tensors
- Validate `input_type` is one of: `"probability"`, `"log_probability"`, `"logits"`
- Validate `reduction` is one of: `"none"`, `"mean"`, `"batchmean"`, `"sum"`
- Check distribution dimension sizes match: `p.size(dim) == q.size(dim)`
- For `pairwise=False`: verify shapes are broadcastable
- For `pairwise=True`: verify `p` and `q` are at least 2D

## Implementation Notes

- All preprocessing (softmax, exp, clamping) happens in C++ kernels for efficiency
- Gradients flow correctly through preprocessing operations
- Follows existing torchscience patterns (see `minkowski_distance`, `histogram`)
