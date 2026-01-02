# Design: torchscience.signal_processing.noise Module

**Date:** 2026-01-02
**Status:** Approved

## Overview

Comprehensive noise generation module for audio/acoustic synthesis, physical simulation, ML augmentation, and general-purpose scientific computing.

## Scope

### Spectral Noise Family

- `white_noise` — flat spectrum (α=0)
- `pink_noise` — 1/f spectrum (α=1) — refactored from existing
- `brown_noise` — 1/f² spectrum (α=2), also known as red/Brownian noise
- `blue_noise` — f spectrum (α=-1)
- `violet_noise` — f² spectrum (α=-2)

### Shot & Impulse Noise

- `poisson_noise` — discrete photon/event counts, tensor rate support
- `shot_noise` — differentiable continuous approximation with Gumbel-softmax relaxation
- `impulse_noise` — salt-and-pepper with independent tensor probabilities

### Out of Scope

- Ornstein-Uhlenbeck process → `torchscience.probability`
- Fractional Brownian motion → `torchscience.probability`

## Module Structure

### Python

```
src/torchscience/signal_processing/noise/
├── __init__.py
├── _white_noise.py
├── _pink_noise.py      # refactored
├── _brown_noise.py
├── _blue_noise.py
├── _violet_noise.py
├── _poisson_noise.py
├── _shot_noise.py
└── _impulse_noise.py
```

### C++ Kernels

Each noise function uses the split CPU/Meta/Autograd/Autocast pattern:

```
src/torchscience/csrc/
├── cpu/signal_processing/noise/
│   ├── white_noise.h
│   ├── pink_noise.h
│   ├── brown_noise.h
│   ├── blue_noise.h
│   ├── violet_noise.h
│   ├── poisson_noise.h
│   ├── shot_noise.h
│   └── impulse_noise.h
├── meta/signal_processing/noise/
│   └── ... (same 8 files)
├── autograd/signal_processing/noise/
│   └── ... (same 8 files)
└── autocast/signal_processing/noise/
    └── ... (same 8 files)
```

### Migration

The existing `composite/signal_processing/noise.h` will be deleted after `pink_noise` is migrated to the split pattern.

## API Design

### Common Signature (Spectral Noise)

All spectral noise functions share the same signature:

```python
def <color>_noise(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> Tensor:
```

- `size`: Shape of output tensor. Last dimension is sample axis; others are batch dimensions.
- Returns tensor with approximately zero mean and unit variance.
- Generation only — users add noise to signals manually (`signal + noise`).

### Spectral Power Laws

| Color  | α (power) | Frequency Scaling | Physical Character |
|--------|-----------|-------------------|-------------------|
| white  | 0         | 1                 | Flat spectrum |
| pink   | 1         | 1/√f              | Equal energy per octave |
| brown  | 2         | 1/f               | Random walk / Brownian |
| blue   | -1        | √f                | High-frequency emphasis |
| violet | -2        | f                 | Differentiated white |

### Poisson Noise (Discrete)

```python
def poisson_noise(
    size: Sequence[int],
    rate: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,  # defaults to torch.int64
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    generator: Optional[Generator] = None,
) -> Tensor:
```

- Returns integer counts drawn from Poisson(rate)
- `rate` can be scalar or tensor (broadcastable with `size`) for spatially-varying rates
- No `requires_grad` — discrete values aren't differentiable
- Default dtype is `int64`, but user can request `float` for convenience

### Shot Noise (Differentiable)

```python
def shot_noise(
    size: Sequence[int],
    rate: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> Tensor:
```

- For `rate >= 10`: Gaussian approximation `N(rate, rate)` — valid and efficient
- For `rate < 10`: Gumbel-softmax relaxation for proper differentiability
- `rate` supports tensor for spatially-varying rates
- Supports `requires_grad` for differentiable pipelines

### Impulse Noise

```python
def impulse_noise(
    size: Sequence[int],
    *,
    p_salt: Union[float, Tensor] = 0.0,
    p_pepper: Union[float, Tensor] = 0.0,
    salt_value: float = 1.0,
    pepper_value: float = -1.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    generator: Optional[Generator] = None,
) -> Tensor:
```

- Independent control over salt and pepper probabilities
- `p_salt` and `p_pepper` support tensor for spatially-varying corruption
- Returns tensor with: `pepper_value` (prob `p_pepper`), `0` (prob `1 - p_salt - p_pepper`), `salt_value` (prob `p_salt`)
- No `requires_grad` — discrete selection isn't differentiable
- Default values (±1) work for normalized signals; override for specific ranges (e.g., 0/255 for images)

## Implementation Strategy

### Kernel Pattern

All noise functions use the split CPU/Meta/Autograd/Autocast pattern for consistency:

| Backend | Responsibility |
|---------|---------------|
| CPU | Core implementation using PyTorch ops (randn, fft, poisson) |
| Meta | Shape inference for lazy tensors |
| Autograd | Gradient computation (where applicable) |
| Autocast | Mixed precision support |

### Spectral Noise Algorithm

All spectral noise types use FFT-based spectral shaping:

1. Generate white noise via `torch.randn`
2. FFT to frequency domain
3. Scale by power law: `1/f^(α/2)` where α varies by color
4. Set DC component to zero (ensures zero mean)
5. IFFT back to time domain
6. Normalize to unit variance

### Shot Noise Algorithm

```
if rate >= 10:
    # Gaussian approximation (fast, accurate for high rates)
    return N(rate, sqrt(rate))
else:
    # Gumbel-softmax relaxation (differentiable for low rates)
    return gumbel_softmax_poisson(rate, temperature=1.0)
```

### Autograd

- Spectral noise: gradients flow through FFT operations when `requires_grad=True`
- `shot_noise`: gradients flow through Gaussian reparameterization or Gumbel-softmax
- `poisson_noise`, `impulse_noise`: no gradients (discrete)

## Testing Strategy

### Statistical Property Tests

| Function | Key Tests |
|----------|-----------|
| `white_noise` | Flat power spectrum, zero mean, unit variance, uncorrelated samples |
| `pink_noise` | 1/f spectrum slope ≈ -1 (log-log), zero mean, unit variance |
| `brown_noise` | 1/f² spectrum slope ≈ -2, zero mean, unit variance |
| `blue_noise` | f spectrum slope ≈ +1, zero mean, unit variance |
| `violet_noise` | f² spectrum slope ≈ +2, zero mean, unit variance |
| `poisson_noise` | Mean ≈ rate, variance ≈ rate, integer values |
| `shot_noise` | Mean ≈ rate, variance ≈ rate, Gumbel-softmax correctness at low rates |
| `impulse_noise` | Correct proportions matching `p_salt`/`p_pepper`, correct values |

### Spectrum Slope Verification

```python
def test_brown_noise_spectrum():
    noise = brown_noise([10000])
    freqs, psd = welch(noise)  # scipy.signal.welch
    # Fit log(psd) vs log(freq), expect slope ≈ -2
    slope = linregress(log(freqs[1:]), log(psd[1:])).slope
    assert -2.3 < slope < -1.7  # tolerance for finite samples
```

### Gradient Tests

- `gradcheck` for spectral noise and `shot_noise` with `requires_grad=True`
- `gradgradcheck` for second-order gradients
- Verify `shot_noise` Gumbel-softmax gradients flow correctly at low rates

### Tensor Parameter Tests

- Verify `poisson_noise`, `shot_noise`, `impulse_noise` work with tensor parameters
- Test broadcasting behavior when parameter shape differs from `size`

### Reproducibility Tests

- Same `generator` seed → identical output
- Different seeds → different output

### Edge Cases

- Empty size, size with zeros, very long sequences
- Extreme parameter values (very low/high rates, `p_salt=0`, `p_pepper=1`)
- Device placement (CPU, CUDA)
- Dtype variations (float32, float64, float16, bfloat16)

## Public API

### Module Exports

```python
# src/torchscience/signal_processing/noise/__init__.py
from ._white_noise import white_noise
from ._pink_noise import pink_noise
from ._brown_noise import brown_noise
from ._blue_noise import blue_noise
from ._violet_noise import violet_noise
from ._poisson_noise import poisson_noise
from ._shot_noise import shot_noise
from ._impulse_noise import impulse_noise

__all__ = [
    "white_noise",
    "pink_noise",
    "brown_noise",
    "blue_noise",
    "violet_noise",
    "poisson_noise",
    "shot_noise",
    "impulse_noise",
]
```

### Documentation Standards

Each function follows NumPy docstring format:

- Mathematical definition section with equations
- Parameters with types and descriptions
- Returns section
- Examples (including reproducibility with generator)
- Notes on spectral properties, normalization, gradient support
- References to relevant papers
- See Also cross-references to related functions

## Dependencies

- **Runtime:** PyTorch only (FFT, randn, poisson)
- **Test:** scipy (for Welch PSD, statistical tests)

## Implementation Order

### Phase 1: Infrastructure & Refactor

1. Refactor `pink_noise` from `CompositeImplicitAutograd` to split CPU/Meta/Autograd/Autocast pattern
2. Delete `composite/signal_processing/noise.h`

### Phase 2: Spectral Noise

3. `white_noise` — establishes the pattern (kernel wrapping `torch.randn`)
4. `brown_noise` — same FFT structure as `pink_noise` with α=2
5. `blue_noise` — FFT structure with α=-1
6. `violet_noise` — FFT structure with α=-2

### Phase 3: Shot & Impulse Noise

7. `poisson_noise` — delegates to `torch.poisson`, tensor rate support
8. `shot_noise` — Gaussian approximation + Gumbel-softmax for low rates
9. `impulse_noise` — masking with independent `p_salt`/`p_pepper` tensor support

### Phase 4: Finalize

10. Update `signal_processing/noise/__init__.py` exports
11. Comprehensive test suite
12. Documentation review
