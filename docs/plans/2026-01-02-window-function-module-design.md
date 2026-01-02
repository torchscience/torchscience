# Window Function Module Design

**Date:** 2026-01-02
**Status:** Approved
**Module:** `torchscience.signal_processing.window_function`

## Overview

Comprehensive window function module providing ~44 functions across 3 phases with:

- **Differentiable parameters** — learn optimal window shapes via autograd
- **Batched generation** — efficient parallel window creation with different parameters
- **Unified API** — consistent signatures, explicit symmetric/periodic variants
- **C++ backend** — CPU/Meta/Autograd kernels following torchscience patterns

## Value Proposition

| Feature | torch.signal.windows | scipy.signal.windows | torchscience |
|---------|---------------------|---------------------|--------------|
| Window count | 9 | 26 | 44 (planned) |
| Differentiable params | No | No | Yes |
| Batched generation | No | No | Yes |
| Unified API | Yes | Yes | Yes |
| GPU support | Yes | No | Yes |

## API Design

### Naming Convention

Each window has two variants:

- **Symmetric:** `{name}_window(n, ...)` — for filter design, FIR coefficients
- **Periodic:** `periodic_{name}_window(n, ...)` — for spectral analysis, STFT

### Signature Patterns

**Parameterless windows:**

```python
def hann_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Returns shape (n,)"""
```

**Parameterized windows:**

```python
def gaussian_window(
    n: int,
    std: Tensor,  # shape () -> (n,), shape (B,) -> (B, n)
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    std: standard deviation parameter (differentiable, batchable)
    requires_grad inherited from std tensor
    """
```

### Batching Semantics

Parameter shape determines output shape:

- `std=torch.tensor(0.4)` (scalar) → window shape `(n,)`
- `std=torch.tensor([0.3, 0.4, 0.5])` (batch) → window shape `(3, n)`

## Phase 1: PyTorch Parity (18 functions)

| Window | Parameters | Symmetric | Periodic |
|--------|------------|-----------|----------|
| Bartlett | none | `bartlett_window` | `periodic_bartlett_window` |
| Blackman | none | `blackman_window` | `periodic_blackman_window` |
| Cosine | none | `cosine_window` | `periodic_cosine_window` |
| Gaussian | `std: Tensor` | `gaussian_window` | `periodic_gaussian_window` |
| Hamming | none | `hamming_window` | `periodic_hamming_window` |
| Hann | none | `hann_window` | `periodic_hann_window` |
| Nuttall | none | `nuttall_window` | `periodic_nuttall_window` |
| General Cosine | `coeffs: Tensor` | `general_cosine_window` | `periodic_general_cosine_window` |
| General Hamming | `alpha: Tensor` | `general_hamming_window` | `periodic_general_hamming_window` |

## Phase 2: Extended Coverage (16 functions)

| Window | Parameters |
|--------|------------|
| Kaiser | `beta: Tensor` |
| Tukey | `alpha: Tensor` |
| Blackman-Harris | none |
| Flat Top | none |
| Bohman | none |
| Parzen | none |
| Lanczos | none |
| Triangular | none |

## Phase 3: Specialized Windows (10 functions)

| Window | Parameters | Notes |
|--------|------------|-------|
| Chebyshev | `attenuation: Tensor` | Equiripple sidelobes |
| DPSS | `bandwidth: Tensor` | Slepian sequences |
| Exponential | `tau: Tensor` | Decay parameter |
| Taylor | `nbar, sll: Tensor` | Radar applications |
| Kaiser-Bessel Derived | `beta: Tensor` | MDCT applications |

## C++ Architecture

### File Structure

```
src/torchscience/csrc/
├── composite/signal_processing/
│   └── window_functions.h          # Dispatch to CPU/Meta, TORCH_LIBRARY_IMPL
├── cpu/signal_processing/
│   └── window_functions.h          # CPU kernels
├── meta/signal_processing/
│   └── window_functions.h          # Shape inference
├── autograd/signal_processing/
│   └── window_functions.h          # Backward for parameterized windows
└── kernel/signal_processing/
    └── window_functions/
        ├── hann.h                   # Kernel math (shared CPU/CUDA)
        ├── gaussian.h
        └── ...
```

### Kernel Pattern

```cpp
// kernel/signal_processing/window_functions/gaussian.h
template<typename scalar_t>
inline scalar_t gaussian_kernel(int64_t i, int64_t n, scalar_t std, bool periodic) {
    scalar_t denom = periodic ? scalar_t(n) : scalar_t(n - 1);
    scalar_t center = denom / scalar_t(2);
    scalar_t x = (scalar_t(i) - center) / (std * center);
    return std::exp(scalar_t(-0.5) * x * x);
}
```

### Single Code Path with Squeeze/Unsqueeze

```cpp
at::Tensor gaussian_window(
    int64_t n,
    const at::Tensor& std_in,  // shape () or (B,)
    bool periodic,
    ...
) {
    // Normalize to batched: () -> (1,)
    bool was_scalar = std_in.dim() == 0;
    at::Tensor std = was_scalar ? std_in.unsqueeze(0) : std_in;

    int64_t batch = std.size(0);
    auto output = at::empty({batch, n}, ...);

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "gaussian_window", [&] {
        auto std_a = std.accessor<scalar_t, 1>();
        auto out_a = output.accessor<scalar_t, 2>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < n; ++i) {
                out_a[b][i] = gaussian_kernel<scalar_t>(i, n, std_a[b], periodic);
            }
        }
    });

    // Squeeze if input was scalar: (1, n) -> (n,)
    return was_scalar ? output.squeeze(0) : output;
}
```

## Python Module Structure

```
src/torchscience/signal_processing/window_function/
├── __init__.py
├── _bartlett_window.py
├── _periodic_bartlett_window.py
├── _blackman_window.py
├── _periodic_blackman_window.py
├── _cosine_window.py
├── _periodic_cosine_window.py
├── _gaussian_window.py
├── _periodic_gaussian_window.py
├── _hamming_window.py
├── _periodic_hamming_window.py
├── _hann_window.py
├── _periodic_hann_window.py
├── _nuttall_window.py
├── _periodic_nuttall_window.py
├── _general_cosine_window.py
├── _periodic_general_cosine_window.py
├── _general_hamming_window.py
├── _periodic_general_hamming_window.py
└── _rectangular_window.py          # existing
```

## Operator Registration

### Schema Definitions

```cpp
TORCH_LIBRARY(torchscience, m) {
    // Parameterless windows (symmetric)
    m.def("bartlett_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("blackman_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("cosine_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("hamming_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("hann_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("nuttall_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");

    // Parameterless windows (periodic)
    m.def("periodic_bartlett_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("periodic_blackman_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("periodic_cosine_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("periodic_hamming_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("periodic_hann_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");
    m.def("periodic_nuttall_window(int n, ScalarType? dtype=None, Layout? layout=None, "
          "Device? device=None, bool requires_grad=False) -> Tensor");

    // Parameterized windows (symmetric)
    m.def("gaussian_window(int n, Tensor std, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");
    m.def("general_cosine_window(int n, Tensor coeffs, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");
    m.def("general_hamming_window(int n, Tensor alpha, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");

    // Parameterized windows (periodic)
    m.def("periodic_gaussian_window(int n, Tensor std, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");
    m.def("periodic_general_cosine_window(int n, Tensor coeffs, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");
    m.def("periodic_general_hamming_window(int n, Tensor alpha, ScalarType? dtype=None, "
          "Layout? layout=None, Device? device=None) -> Tensor");
}
```

### Backend Registration

```cpp
// CompositeExplicitAutograd for parameterless (no tensor inputs)
TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, m) {
    m.impl("hann_window", &torchscience::window_function::hann_window);
    m.impl("periodic_hann_window", &torchscience::window_function::periodic_hann_window);
    // ... other parameterless windows
}

// CPU for parameterized windows
TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("gaussian_window", &torchscience::cpu::gaussian_window);
    m.impl("periodic_gaussian_window", &torchscience::cpu::periodic_gaussian_window);
    // ...
}

// Meta for shape inference
TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("gaussian_window", &torchscience::meta::gaussian_window);
    // ...
}

// Autograd for differentiable parameters
TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gaussian_window", &torchscience::autograd::gaussian_window);
    // ...
}
```

## Testing Strategy

### Test Infrastructure

```python
class WindowOpDescriptor:
    name: str
    func: Callable
    reference_func: Callable  # Python reference implementation
    pytorch_func: Optional[Callable]  # torch.signal.windows.* if available
    parameters: List[str]  # e.g., ["std"] for gaussian
    expected_values: List[ExpectedValue]
    analytical_derivatives: Dict[str, Callable]  # param -> derivative func
    is_periodic: bool
    supported_dtypes: List[torch.dtype]
    tolerances: ToleranceConfig

class WindowOpTestCase(TestCase):
    """Base class for window function tests."""

    # Inherited tests (run automatically)
    def test_basic_correctness(self): ...
    def test_pytorch_comparison(self): ...
    def test_symmetry(self): ...
    def test_dtype_support(self): ...
    def test_device_support(self): ...
    def test_meta_tensor(self): ...
    def test_empty_window(self): ...
    def test_single_element(self): ...
    def test_large_window(self): ...
    def test_torch_compile(self): ...

    # Parameterized window tests
    def test_batched_parameters(self): ...
    def test_gradcheck(self): ...
    def test_gradgradcheck(self): ...
    def test_analytical_derivatives(self): ...

    # Frequency domain tests
    def test_coherent_gain(self): ...
    def test_parseval_energy(self): ...
```

### Test Coverage

- **Correctness:** Compare against Python reference implementation
- **PyTorch comparison:** Validate against `torch.signal.windows.*` where available
- **Autograd:** `gradcheck`, `gradgradcheck`, explicit analytical derivative tests
- **Batching:** Verify shape semantics for scalar vs batched parameters
- **Edge cases:** n=0, n=1, large n, extreme parameter values
- **Frequency domain:** Coherent gain, Parseval energy conservation

## Mathematical Definitions

### Hann Window (Symmetric)

```
w[k] = 0.5 * (1 - cos(2πk / (n-1))),  for k = 0, 1, ..., n-1
```

### Hann Window (Periodic)

```
w[k] = 0.5 * (1 - cos(2πk / n)),  for k = 0, 1, ..., n-1
```

### Gaussian Window

```
w[k] = exp(-0.5 * ((k - (n-1)/2) / (std * (n-1)/2))^2)
```

### General Cosine Window

```
w[k] = sum_{i=0}^{M-1} (-1)^i * a_i * cos(2πik / (n-1))
```

### General Hamming Window

```
w[k] = alpha - (1 - alpha) * cos(2πk / (n-1))
```

## Implementation Order

### Phase 1 Priority

1. `hann_window` / `periodic_hann_window` — most common
2. `hamming_window` / `periodic_hamming_window` — second most common
3. `blackman_window` / `periodic_blackman_window` — 3-term cosine
4. `bartlett_window` / `periodic_bartlett_window` — triangular
5. `cosine_window` / `periodic_cosine_window` — half-cosine
6. `nuttall_window` / `periodic_nuttall_window` — 4-term cosine
7. `gaussian_window` / `periodic_gaussian_window` — first parameterized
8. `general_hamming_window` / `periodic_general_hamming_window` — parameterized
9. `general_cosine_window` / `periodic_general_cosine_window` — N-term parameterized

## References

- Harris, F.J. "On the use of windows for harmonic analysis with the discrete Fourier transform," Proceedings of the IEEE, vol. 66, no. 1, pp. 51-83, Jan. 1978.
- Oppenheim, A.V. and Schafer, R.W. "Discrete-Time Signal Processing," 3rd ed., Prentice Hall, 2009.
- [PyTorch torch.signal.windows](https://docs.pytorch.org/docs/stable/signal.html)
- [SciPy scipy.signal.windows](https://docs.scipy.org/doc/scipy/reference/signal.windows.html)
