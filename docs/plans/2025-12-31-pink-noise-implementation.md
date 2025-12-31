# Pink Noise Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.signal_processing.noise.pink_noise` - a batched pink noise generator with 1/f power spectrum using spectral shaping.

**Architecture:** Use CompositeImplicitAutograd to leverage ATen's FFT ops (`randn`, `fft_rfft`, `fft_irfft`) which handle CPU/CUDA dispatch automatically. Gradients flow through FFT ops naturally.

**Tech Stack:** C++ ATen ops, PyTorch FFT, CompositeImplicitAutograd dispatch

---

## Task 1: Register operator schema

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp:189-191`

**Step 1: Add schema definition**

Add after line 189 (after floyd_warshall):

```cpp
  // signal_processing.noise
  module.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
```

**Step 2: Verify build**

Run: `uv run python -c "import torchscience; print('OK')"`
Expected: OK (schema registered, no implementation yet)

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(schema): register pink_noise operator"
```

---

## Task 2: Implement composite operator

**Files:**
- Create: `src/torchscience/csrc/composite/signal_processing/noise.h`

**Step 1: Write the implementation**

```cpp
#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>

namespace torchscience::noise {

inline at::Tensor pink_noise(
  at::IntArrayRef size,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad,
  const c10::optional<at::Generator> generator
) {
  TORCH_CHECK(size.size() > 0, "pink_noise: size must be non-empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "pink_noise: size elements must be non-negative, got ", s);
  }

  // Determine output dtype and device
  at::ScalarType out_dtype = dtype.value_or(
    c10::typeMetaToScalarType(at::get_default_dtype())
  );
  at::Device out_device = device.value_or(at::kCPU);
  at::Layout out_layout = layout.value_or(at::kStrided);

  // Handle empty tensor case
  int64_t n = size.back();  // Last dimension is the sample axis
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }

  if (numel == 0 || n == 0) {
    auto options = at::TensorOptions()
      .dtype(out_dtype)
      .layout(out_layout)
      .device(out_device)
      .requires_grad(requires_grad);
    return at::empty(size.vec(), options);
  }

  // Handle n=1 special case (return zeros - can't have 1/f spectrum with 1 sample)
  if (n == 1) {
    auto options = at::TensorOptions()
      .dtype(out_dtype)
      .layout(out_layout)
      .device(out_device)
      .requires_grad(requires_grad);
    return at::zeros(size.vec(), options);
  }

  // Use float64 for computation to ensure precision
  at::ScalarType compute_dtype = at::kFloat64;

  // For half-precision types, we'll compute in float32 then cast back
  if (out_dtype == at::kHalf || out_dtype == at::kBFloat16) {
    compute_dtype = at::kFloat32;
  } else if (out_dtype == at::kFloat64) {
    compute_dtype = at::kFloat64;
  } else {
    compute_dtype = at::kFloat32;
  }

  auto compute_options = at::TensorOptions()
    .dtype(compute_dtype)
    .layout(out_layout)
    .device(out_device);

  // Step 1: Generate white noise
  at::Tensor white = at::randn(size.vec(), generator, compute_options);

  // Step 2: Compute real FFT along last dimension
  at::Tensor spectrum = at::fft_rfft(white, /*n=*/c10::nullopt, /*dim=*/-1);

  // Step 3: Build frequency scaling vector S[k] = sqrt(N/k) for k > 0, S[0] = 0
  // spectrum has shape [..., N/2 + 1]
  int64_t freq_size = spectrum.size(-1);

  // Create frequency indices [0, 1, 2, ..., N/2]
  at::Tensor freq_indices = at::arange(freq_size, compute_options);

  // Scaling: sqrt(N/k) for k > 0, 0 for k = 0
  // We use sqrt(N) / sqrt(k) = sqrt(N/k)
  at::Tensor scaling = at::where(
    freq_indices > 0,
    at::sqrt(static_cast<double>(n) / freq_indices),
    at::zeros({1}, compute_options)
  );

  // Step 4: Apply scaling (broadcast over batch dimensions)
  at::Tensor shaped_spectrum = spectrum * scaling;

  // Step 5: Inverse FFT to get time-domain signal
  at::Tensor result = at::fft_irfft(shaped_spectrum, /*n=*/n, /*dim=*/-1);

  // Step 6: Normalize to approximately unit variance
  // The theoretical variance before normalization is sum(N/k) for k=1..N/2
  // which is approximately N * (ln(N/2) + gamma) where gamma is Euler's constant
  // We normalize by sqrt of this
  double harmonic_sum = 0.0;
  for (int64_t k = 1; k <= n / 2; ++k) {
    harmonic_sum += static_cast<double>(n) / static_cast<double>(k);
  }
  double norm_factor = std::sqrt(harmonic_sum / static_cast<double>(n));

  result = result / norm_factor;

  // Cast to output dtype if needed
  if (result.scalar_type() != out_dtype) {
    result = result.to(out_dtype);
  }

  // Set requires_grad if requested
  if (requires_grad) {
    result = result.requires_grad_(true);
  }

  return result;
}

} // namespace torchscience::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, module) {
  module.impl("pink_noise", &torchscience::noise::pink_noise);
}
```

**Step 2: Include in build**

Add to `src/torchscience/csrc/torchscience.cpp` at the top includes section:

```cpp
#include "composite/signal_processing/noise.h"
```

**Step 3: Verify build and basic function**

Run: `uv run python -c "import torch; import torchscience._csrc; print(torch.ops.torchscience.pink_noise([100]).shape)"`
Expected: `torch.Size([100])`

**Step 4: Commit**

```bash
git add src/torchscience/csrc/composite/signal_processing/noise.h
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(cpu): implement pink_noise composite operator"
```

---

## Task 3: Add Python wrapper

**Files:**
- Create: `src/torchscience/signal_processing/noise/_pink_noise.py`
- Modify: `src/torchscience/signal_processing/noise/__init__.py`
- Modify: `src/torchscience/signal_processing/__init__.py`

**Step 1: Write Python wrapper**

Create `src/torchscience/signal_processing/noise/_pink_noise.py`:

```python
from typing import Optional, Sequence

import torch
from torch import Generator, Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def pink_noise(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> Tensor:
    """
    Generate pink noise with 1/f power spectral density.

    Pink noise (also known as 1/f noise or flicker noise) has a power spectral
    density that is inversely proportional to frequency. This gives it equal
    energy per octave, making it useful for audio synthesis, testing, and
    scientific simulations.

    Mathematical Definition
    -----------------------
    The power spectral density S(f) of pink noise follows:

        S(f) ~ 1/f

    This is generated using spectral shaping:
    1. Generate white noise w
    2. Transform to frequency domain: W = FFT(w)
    3. Apply 1/sqrt(f) scaling: P = W / sqrt(f)
    4. Transform back: p = IFFT(P)
    5. Normalize to unit variance

    Parameters
    ----------
    size : Sequence[int]
        Shape of the output tensor. The last dimension is treated as the
        time/sample axis where the 1/f spectrum is applied. Other dimensions
        are batch dimensions generating independent noise sequences.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor. If None, uses the
        default floating point type.
    layout : torch.layout, optional
        The desired layout of the returned tensor. Default: torch.strided.
    device : torch.device, optional
        The desired device of the returned tensor. Default: CPU.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients. Default: False.
    generator : torch.Generator, optional
        A pseudorandom number generator for sampling. If None, uses the
        default generator.

    Returns
    -------
    Tensor
        A tensor of shape `size` containing pink noise samples with
        approximately zero mean and unit variance.

    Examples
    --------
    Generate 1D pink noise with 1000 samples:

    >>> noise = pink_noise([1000])
    >>> noise.shape
    torch.Size([1000])

    Generate batched pink noise (4 channels, 1000 samples each):

    >>> noise = pink_noise([4, 1000])
    >>> noise.shape
    torch.Size([4, 1000])

    Generate reproducible noise using a generator:

    >>> g = torch.Generator().manual_seed(42)
    >>> noise1 = pink_noise([100], generator=g)
    >>> g = torch.Generator().manual_seed(42)
    >>> noise2 = pink_noise([100], generator=g)
    >>> torch.allclose(noise1, noise2)
    True

    Generate on GPU with gradients:

    >>> noise = pink_noise([1000], device='cuda', requires_grad=True)  # doctest: +SKIP

    Raises
    ------
    RuntimeError
        If size is empty or contains negative values.

    See Also
    --------
    torch.randn : Generate white (Gaussian) noise

    Notes
    -----
    Spectral Properties
    ^^^^^^^^^^^^^^^^^^^
    The output has a power spectrum proportional to 1/f, meaning:
    - Lower frequencies have more power than higher frequencies
    - Equal power per octave (logarithmic frequency bands)
    - The DC component (f=0) is set to zero to ensure zero mean

    Normalization
    ^^^^^^^^^^^^^
    The output is normalized to have approximately unit variance,
    similar to torch.randn. This is achieved by dividing by the
    theoretical standard deviation of the shaped noise.

    Gradient Support
    ^^^^^^^^^^^^^^^^
    When requires_grad=True, gradients flow through the FFT operations.
    This enables use in differentiable audio synthesis and learned noise
    models.

    References
    ----------
    N. J. Kasdin, "Discrete simulation of colored noise and stochastic
    processes and 1/f^alpha power law noise generation," Proceedings of
    the IEEE, vol. 83, no. 5, pp. 802-827, 1995.

    J. Timmer and M. Koenig, "On generating power law noise," Astronomy
    and Astrophysics, vol. 300, pp. 707-710, 1995.
    """
    return torch.ops.torchscience.pink_noise(
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
```

**Step 2: Update noise/__init__.py**

```python
from ._pink_noise import pink_noise

__all__ = [
    "pink_noise",
]
```

**Step 3: Update signal_processing/__init__.py**

Add `noise` to the imports:

```python
from . import filter, noise, transform, waveform, window_function

__all__ = [
    "filter",
    "noise",
    "transform",
    "waveform",
    "window_function",
]
```

**Step 4: Verify import**

Run: `uv run python -c "from torchscience.signal_processing.noise import pink_noise; print(pink_noise([10]))"`
Expected: A tensor of shape [10] with pink noise values

**Step 5: Commit**

```bash
git add src/torchscience/signal_processing/noise/_pink_noise.py
git add src/torchscience/signal_processing/noise/__init__.py
git add src/torchscience/signal_processing/__init__.py
git commit -m "feat(python): add pink_noise Python wrapper"
```

---

## Task 4: Add basic tests

**Files:**
- Create: `tests/torchscience/signal_processing/noise/__init__.py`
- Create: `tests/torchscience/signal_processing/noise/test__pink_noise.py`

**Step 1: Create test directory init**

Create empty `tests/torchscience/signal_processing/noise/__init__.py`

**Step 2: Write tests**

Create `tests/torchscience/signal_processing/noise/test__pink_noise.py`:

```python
import math

import pytest
import torch
import torch.testing

from torchscience.signal_processing.noise import pink_noise


class TestPinkNoiseShape:
    """Tests for output shape correctness."""

    def test_1d_shape(self):
        """Test 1D output shape."""
        result = pink_noise([100])
        assert result.shape == torch.Size([100])

    def test_2d_shape(self):
        """Test 2D (batched) output shape."""
        result = pink_noise([4, 100])
        assert result.shape == torch.Size([4, 100])

    def test_3d_shape(self):
        """Test 3D (batch, channels, samples) output shape."""
        result = pink_noise([2, 3, 100])
        assert result.shape == torch.Size([2, 3, 100])

    def test_empty_last_dim(self):
        """Test empty tensor when last dim is 0."""
        result = pink_noise([10, 0])
        assert result.shape == torch.Size([10, 0])
        assert result.numel() == 0

    def test_empty_batch_dim(self):
        """Test empty tensor when batch dim is 0."""
        result = pink_noise([0, 100])
        assert result.shape == torch.Size([0, 100])
        assert result.numel() == 0

    def test_single_sample(self):
        """Test n=1 returns zeros."""
        result = pink_noise([1])
        assert result.shape == torch.Size([1])
        torch.testing.assert_close(
            result, torch.zeros(1), rtol=0, atol=0
        )

    def test_two_samples(self):
        """Test minimal case n=2."""
        result = pink_noise([2])
        assert result.shape == torch.Size([2])


class TestPinkNoiseDtype:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float64,
    ])
    def test_standard_dtypes(self, dtype):
        """Test standard floating point dtypes."""
        result = pink_noise([100], dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.bfloat16,
    ])
    def test_half_precision_dtypes(self, dtype):
        """Test half-precision dtypes."""
        result = pink_noise([100], dtype=dtype)
        assert result.dtype == dtype

    def test_default_dtype(self):
        """Test default dtype is float32."""
        result = pink_noise([100])
        assert result.dtype == torch.float32


class TestPinkNoiseDevice:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test CPU device."""
        result = pink_noise([100], device='cpu')
        assert result.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device."""
        result = pink_noise([100], device='cuda')
        assert result.device.type == 'cuda'


class TestPinkNoiseStatistics:
    """Tests for statistical properties."""

    def test_approximately_zero_mean(self):
        """Test output has approximately zero mean."""
        torch.manual_seed(42)
        result = pink_noise([10000], dtype=torch.float64)
        mean = result.mean().item()
        assert abs(mean) < 0.1, f"Mean {mean} too far from 0"

    def test_approximately_unit_variance(self):
        """Test output has approximately unit variance."""
        torch.manual_seed(42)
        result = pink_noise([10000], dtype=torch.float64)
        var = result.var().item()
        assert 0.5 < var < 2.0, f"Variance {var} not near 1"

    def test_batched_independence(self):
        """Test different batch elements are independent."""
        torch.manual_seed(42)
        result = pink_noise([100, 1000], dtype=torch.float64)

        # Compute correlation between first two batch elements
        x = result[0] - result[0].mean()
        y = result[1] - result[1].mean()
        corr = (x * y).sum() / (x.norm() * y.norm())

        # Should be near zero (independent)
        assert abs(corr.item()) < 0.2, f"Correlation {corr} too high"


class TestPinkNoiseSpectrum:
    """Tests for spectral properties."""

    def test_power_spectrum_slope(self):
        """Test that power spectrum follows 1/f (slope -1 on log-log)."""
        torch.manual_seed(42)
        n = 4096
        result = pink_noise([n], dtype=torch.float64)

        # Compute power spectrum
        spectrum = torch.fft.rfft(result)
        power = torch.abs(spectrum) ** 2

        # Fit slope on log-log scale (exclude DC and high frequencies)
        freq_start = 10
        freq_end = n // 4
        freqs = torch.arange(freq_start, freq_end, dtype=torch.float64)
        powers = power[freq_start:freq_end]

        # Log-log fit: log(P) = slope * log(f) + intercept
        log_f = torch.log(freqs)
        log_p = torch.log(powers)

        # Linear regression
        n_pts = len(freqs)
        slope = (n_pts * (log_f * log_p).sum() - log_f.sum() * log_p.sum()) / \
                (n_pts * (log_f ** 2).sum() - log_f.sum() ** 2)

        # Slope should be approximately -1 for 1/f noise
        assert -1.5 < slope.item() < -0.5, f"Slope {slope} not near -1"

    def test_dc_component_near_zero(self):
        """Test that DC component is near zero (zero mean)."""
        torch.manual_seed(42)
        result = pink_noise([1000], dtype=torch.float64)

        spectrum = torch.fft.rfft(result)
        dc = spectrum[0].abs().item()

        # DC should be small relative to other components
        assert dc < 10, f"DC component {dc} too large"


class TestPinkNoiseReproducibility:
    """Tests for reproducibility with generators."""

    def test_generator_reproducibility(self):
        """Test same generator seed gives same output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = pink_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(42)
        result2 = pink_noise([100], generator=g2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_different_output(self):
        """Test different seeds give different output."""
        g1 = torch.Generator().manual_seed(42)
        result1 = pink_noise([100], generator=g1)

        g2 = torch.Generator().manual_seed(43)
        result2 = pink_noise([100], generator=g2)

        assert not torch.allclose(result1, result2)


class TestPinkNoiseGradient:
    """Tests for gradient support."""

    def test_requires_grad_propagates(self):
        """Test requires_grad parameter works."""
        result = pink_noise([100], requires_grad=True)
        assert result.requires_grad

    def test_gradient_flows(self):
        """Test gradients can be computed."""
        result = pink_noise([100], dtype=torch.float64, requires_grad=True)
        loss = result.sum()
        loss.backward()
        # Should not raise


class TestPinkNoiseCompile:
    """Tests for torch.compile compatibility."""

    def test_basic_compile(self):
        """Test basic torch.compile works."""
        compiled = torch.compile(pink_noise)
        result = compiled([100])
        assert result.shape == torch.Size([100])

    def test_compile_matches_eager(self):
        """Test compiled output matches eager mode."""
        g1 = torch.Generator().manual_seed(42)
        eager = pink_noise([100], generator=g1)

        compiled = torch.compile(pink_noise)
        g2 = torch.Generator().manual_seed(42)
        compiled_result = compiled([100], generator=g2)

        torch.testing.assert_close(eager, compiled_result)


class TestPinkNoiseEdgeCases:
    """Tests for edge cases."""

    def test_large_tensor(self):
        """Test with large tensor."""
        result = pink_noise([100000])
        assert result.shape == torch.Size([100000])

    def test_small_tensor(self):
        """Test with small tensor (n=2)."""
        result = pink_noise([2])
        assert result.shape == torch.Size([2])
        assert torch.isfinite(result).all()

    def test_contiguous_output(self):
        """Test output is contiguous."""
        result = pink_noise([100])
        assert result.is_contiguous()
```

**Step 3: Run tests**

Run: `uv run pytest tests/torchscience/signal_processing/noise/test__pink_noise.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/torchscience/signal_processing/noise/__init__.py
git add tests/torchscience/signal_processing/noise/test__pink_noise.py
git commit -m "test(pink_noise): add comprehensive tests"
```

---

## Task 5: Update design document status

**Files:**
- Modify: `docs/plans/2025-12-31-pink-noise-design.md`

**Step 1: Mark as implemented**

Change line 3 from:
```
**Status:** Complete
```
to:
```
**Status:** Implemented
```

**Step 2: Commit**

```bash
git add docs/plans/2025-12-31-pink-noise-design.md
git commit -m "docs: mark pink_noise design as implemented"
```

---

## Final Verification

Run full test suite for noise module:

```bash
uv run pytest tests/torchscience/signal_processing/noise/ -v
```

Expected: All tests pass

Run quick smoke test:

```bash
uv run python -c "
import torch
from torchscience.signal_processing.noise import pink_noise

# Basic generation
x = pink_noise([1000])
print(f'Shape: {x.shape}')
print(f'Mean: {x.mean():.4f}')
print(f'Std: {x.std():.4f}')

# Batched
x = pink_noise([4, 1000])
print(f'Batched shape: {x.shape}')

# With gradients
x = pink_noise([100], requires_grad=True)
x.sum().backward()
print('Gradients: OK')

# Reproducibility
g = torch.Generator().manual_seed(42)
x1 = pink_noise([100], generator=g)
g = torch.Generator().manual_seed(42)
x2 = pink_noise([100], generator=g)
print(f'Reproducible: {torch.allclose(x1, x2)}')

print('All checks passed!')
"
```
