# Design: `torchscience.signal_processing.noise.pink_noise`

**Status:** Implemented
**Date:** 2025-12-31

## Overview

A general-purpose pink noise generator with batched generation support, exact 1/f power spectrum via spectral shaping, and full autograd compatibility.

## API

```python
def pink_noise(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
```

### Parameters

- `size` – Shape of output tensor (e.g., `(1000,)` for 1D, `(batch, channels, samples)` for batched). The last dimension is the time/sample axis where the 1/f spectrum is applied.
- `dtype` – Output dtype (default: default float type)
- `layout` – Tensor layout (default: strided)
- `device` – Target device (cpu, cuda, etc.)
- `requires_grad` – Enable gradient tracking
- `generator` – Optional RNG for reproducibility

### Returns

Tensor of shape `size` containing pink noise with 1/f power spectral density.

### Behavior

- Each "row" along the last dimension is an independent pink noise sequence
- Output is normalized to have approximately unit variance (matching `torch.randn` convention)
- The DC component (frequency 0) is set to zero to ensure zero mean

## Algorithm (Spectral Method)

The spectral method generates pink noise by shaping white noise in the frequency domain:

### Forward Pass

1. Generate white noise `w` of shape `size` using `torch.randn(..., generator=generator)`
2. Compute real FFT along the last dimension: `W = rfft(w)`
3. Build frequency scaling vector `S[k] = 1/sqrt(f[k])` for k > 0, with `S[0] = 0` (zero DC)
4. Apply scaling: `P = W * S`
5. Inverse real FFT: `p = irfft(P)`
6. Normalize to unit variance

### Frequency Vector

For N samples, frequencies are `f[k] = k/N` for k = 0, 1, ..., N/2. The scaling is `S[k] = 1/sqrt(k/N) = sqrt(N/k)` for k > 0.

### Normalization

The raw output has variance depending on N. We normalize by the theoretical factor `sqrt(sum(1/k))` for k=1..N/2, which is approximately `sqrt(ln(N/2) + gamma)` where gamma is Euler's constant. This ensures output variance ≈ 1 regardless of sequence length.

### Autograd

Gradients flow naturally through `rfft`, multiplication, and `irfft`. The scaling vector is a constant (no learnable parameters).

## C++ Backend Structure

### Operator Schema

In `torchscience.cpp`:

```cpp
m.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
```

### File Structure

```
src/torchscience/csrc/
├── cpu/signal_processing/noise/pink_noise.h      # CPU kernel
├── meta/signal_processing/noise/pink_noise.h     # Shape inference
├── cuda/signal_processing/noise/pink_noise.cu    # CUDA kernel
└── torchscience.cpp                              # Schema registration
```

### CPU Implementation

- Uses ATen's `at::randn` with generator support
- Uses `at::fft_rfft` and `at::fft_irfft` for FFT operations
- Builds scaling tensor, applies element-wise multiply
- Registers via `TORCH_LIBRARY_IMPL(torchscience, CPU, m)`

### Meta Implementation

- Returns empty tensor with correct shape, dtype, device
- No actual computation, just shape inference for tracing/compile
- Registers via `TORCH_LIBRARY_IMPL(torchscience, Meta, m)`

### CUDA Implementation

- Same algorithm as CPU but using CUDA tensors
- ATen FFT ops automatically dispatch to cuFFT
- Registers via `TORCH_LIBRARY_IMPL(torchscience, CUDA, m)`

## Python Wrapper

Location: `src/torchscience/signal_processing/noise/_pink_noise.py`

```python
def pink_noise(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Docstring with mathematical definition, parameters, examples..."""
    return torch.ops.torchscience.pink_noise(
        size, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, generator=generator,
    )
```

### Module Exports

- Export from `noise/__init__.py`
- Add `noise` to `signal_processing/__init__.py`

## Testing

Location: `tests/torchscience/signal_processing/noise/test__pink_noise.py`

1. **Shape tests** – Verify output shape matches input `size` for 1D, 2D, batched
2. **Dtype/device tests** – Verify output dtype and device match parameters
3. **Statistical tests** – Verify approximately zero mean, unit variance
4. **Spectral tests** – Verify power spectrum follows 1/f (linear on log-log plot with slope -1)
5. **Reproducibility tests** – Same generator seed produces identical output
6. **Gradient tests** – `torch.autograd.gradcheck` with `requires_grad=True`
7. **Independence tests** – Different batch elements are uncorrelated

## Error Handling & Edge Cases

### Input Validation

- `size` must be non-empty sequence of non-negative integers
- If any dimension is 0, return empty tensor with that shape
- Last dimension (sample axis) must be >= 1 for meaningful noise

### Edge Cases

| Case | Behavior |
|------|----------|
| `size=(0,)` | Return empty tensor shape `(0,)` |
| `size=(N, 0)` | Return empty tensor shape `(N, 0)` |
| `size=(1,)` | Single sample, return 0 (zero mean, can't have 1/f spectrum) |
| `size=(2,)` | Minimal case, still apply algorithm |

### Numerical Considerations

- For very short sequences (N < 10), the 1/f approximation is coarse but valid
- Scaling vector computed in float64 internally for precision, then cast to output dtype
- Half-precision (float16, bfloat16) supported via promotion during FFT, then cast back

### Error Messages

- Clear messages for invalid size (negative dimensions, non-integer)
- Device mismatch between generator and requested device raises informative error
