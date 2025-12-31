# T-Test Implementation Design

## Status: Complete

## Overview

Implement three t-test functions in `torchscience.statistics.hypothesis_test`:

- `one_sample_t_test` - Tests if sample mean differs from a known population mean
- `two_sample_t_test` - Tests if two independent samples have different means
- `paired_t_test` - Tests if paired/matched samples have different means

## API Design

### Function Signatures

```python
def one_sample_t_test(
    input: Tensor,
    popmean: float = 0.0,
    *,
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> tuple[Tensor, Tensor, Tensor]:
    """Test if sample mean differs from a known population mean."""
    ...

def two_sample_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    equal_var: bool = False,  # Default to Welch's (safer)
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    """Test if two independent samples have different means."""
    ...

def paired_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    """Test if paired samples have different means."""
    ...
```

### Return Value

All functions return `(statistic, pvalue, df)` as plain tuples of tensors with shape `(*batch,)`.

### Batching Convention

Last dimension contains samples. Input shape `(*batch, n_samples)` produces results of shape `(*batch,)`.

### Dtype Handling

Inputs promoted to common float type. Half-precision computes internally in float32 for numerical stability.

## Mathematical Formulas

### One-Sample T-Test

```
t = (x̄ - μ₀) / (s / √n)
df = n - 1
```

Where:
- `x̄` = sample mean
- `μ₀` = hypothesized population mean
- `s` = sample standard deviation
- `n` = sample size

### Two-Sample T-Test (Equal Variance / Student's)

```
t = (x̄₁ - x̄₂) / (s_p * √(1/n₁ + 1/n₂))
s_p = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2)]
df = n₁ + n₂ - 2
```

### Two-Sample T-Test (Unequal Variance / Welch's)

```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

Note: Welch's df is non-integer (Welch-Satterthwaite approximation).

### Paired T-Test

Reduces to one-sample t-test on differences `d = x₁ - x₂`.

### P-Value Computation

Using `incomplete_beta`:

```
x = df / (df + t²)
p_one_tail = 0.5 * incomplete_beta(x, df/2, 0.5)
```

Alternative hypothesis handling:
- `"two-sided"`: `p = 2 * p_one_tail`
- `"less"`: `p = p_one_tail` if `t < 0` else `1 - p_one_tail`
- `"greater"`: `p = p_one_tail` if `t > 0` else `1 - p_one_tail`

## Implementation Structure

### C++ File Organization

```
src/torchscience/csrc/
├── kernel/statistics/hypothesis_test/
│   ├── t_test_common.h         # Shared: t_cdf, p-value from t
│   ├── one_sample_t_test.h     # Forward kernel
│   ├── two_sample_t_test.h     # Forward kernel
│   └── paired_t_test.h         # Forward kernel (delegates to one_sample)
├── cpu/statistics/hypothesis_test/
│   ├── one_sample_t_test.h     # CPU dispatch + TORCH_LIBRARY_IMPL
│   ├── two_sample_t_test.h
│   └── paired_t_test.h
├── meta/statistics/hypothesis_test/
│   ├── one_sample_t_test.h     # Shape inference
│   ├── two_sample_t_test.h
│   └── paired_t_test.h
└── torchscience.cpp            # Schema registration
```

### Schema Registration

In `torchscience.cpp`:

```cpp
m.def("one_sample_t_test(Tensor input, float popmean, str alternative) -> (Tensor, Tensor, Tensor)");
m.def("two_sample_t_test(Tensor input1, Tensor input2, bool equal_var, str alternative) -> (Tensor, Tensor, Tensor)");
m.def("paired_t_test(Tensor input1, Tensor input2, str alternative) -> (Tensor, Tensor, Tensor)");
```

### Autograd

No autograd wrapper needed. T-tests return statistics/p-values - users don't typically backprop through hypothesis tests. If needed later, autograd can be added since `incomplete_beta` already supports it.

## Python Module Structure

```
src/torchscience/statistics/hypothesis_test/
├── __init__.py                 # Exports all three functions
├── _one_sample_t_test.py
├── _two_sample_t_test.py
└── _paired_t_test.py
```

Each Python file is a thin wrapper that validates the `alternative` parameter and dispatches to `torch.ops.torchscience.*`.

Docstrings follow NumPy style with:
- Mathematical definition
- Parameters / Returns
- Examples (basic usage, batched usage, one-sided tests)
- Notes on NaN handling (insufficient samples, zero variance)
- See Also referencing scipy.stats equivalents

## Testing Strategy

Test file: `tests/torchscience/statistics/hypothesis_test/test__t_test.py`

### Test Categories

1. **Correctness vs scipy** - Compare against `scipy.stats.ttest_1samp`, `ttest_ind`, `ttest_rel`
   - Normal random data
   - Known edge cases (identical values, large differences)
   - All three `alternative` options

2. **Batched computation** - Verify `(*batch, n_samples)` produces correct `(*batch,)` output
   - Single batch dim: `(5, 100)` -> `(5,)`
   - Multiple batch dims: `(2, 3, 100)` -> `(2, 3,)`

3. **Edge cases / NaN handling**
   - `n < 2` -> NaN (insufficient samples)
   - Zero variance -> NaN (division by zero)
   - Welch's with `n₁=1` or `n₂=1` -> NaN

4. **Dtype coverage** - float32, float64 (skip float16/bfloat16 - hypothesis tests need precision)

5. **Equal vs unequal variance** - Verify `equal_var=True` gives different results than `equal_var=False` when variances differ

6. **Device coverage** - CPU (CUDA if available)

## Dependencies

- `incomplete_beta` - Already implemented, provides t-distribution CDF computation

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API style | Separate functions | Clearer intent, better type signatures |
| Return type | Plain tuple | Simple, matches PyTorch conventions |
| Welch's default | `equal_var=False` | Safer, doesn't assume equal variances |
| Batching | Last dim = samples | Standard PyTorch convention |
| Backend | C++ | Consistent with existing operators |
| Autograd | Not implemented | Rarely needed for hypothesis tests |

## Files to Create/Modify

### New Files
- `src/torchscience/csrc/kernel/statistics/hypothesis_test/t_test_common.h`
- `src/torchscience/csrc/kernel/statistics/hypothesis_test/one_sample_t_test.h`
- `src/torchscience/csrc/kernel/statistics/hypothesis_test/two_sample_t_test.h`
- `src/torchscience/csrc/kernel/statistics/hypothesis_test/paired_t_test.h`
- `src/torchscience/csrc/cpu/statistics/hypothesis_test/one_sample_t_test.h`
- `src/torchscience/csrc/cpu/statistics/hypothesis_test/two_sample_t_test.h`
- `src/torchscience/csrc/cpu/statistics/hypothesis_test/paired_t_test.h`
- `src/torchscience/csrc/meta/statistics/hypothesis_test/one_sample_t_test.h`
- `src/torchscience/csrc/meta/statistics/hypothesis_test/two_sample_t_test.h`
- `src/torchscience/csrc/meta/statistics/hypothesis_test/paired_t_test.h`
- `src/torchscience/statistics/hypothesis_test/_one_sample_t_test.py`
- `src/torchscience/statistics/hypothesis_test/_two_sample_t_test.py`
- `src/torchscience/statistics/hypothesis_test/_paired_t_test.py`
- `tests/torchscience/statistics/hypothesis_test/__init__.py`
- `tests/torchscience/statistics/hypothesis_test/test__t_test.py`

### Modified Files
- `src/torchscience/csrc/torchscience.cpp` - Add schema definitions
- `src/torchscience/statistics/hypothesis_test/__init__.py` - Export functions
- `src/torchscience/statistics/__init__.py` - Export hypothesis_test submodule
