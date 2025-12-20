# torchscience

Scientific computing operators for PyTorch with full autograd, torch.compile, and vmap support.

## Installation

```bash
pip install torchscience
```

For development:

```bash
git clone https://github.com/0x00b1/torchscience.git
cd torchscience
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import torchscience

# Special functions with automatic differentiation
z = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
y = torchscience.special_functions.gamma(z)
y.sum().backward()
print(z.grad)

# Works with torch.compile
compiled_gamma = torch.compile(torchscience.special_functions.gamma)

# Works with vmap for batching
batched_gamma = torch.func.vmap(torchscience.special_functions.gamma)

# Signal processing
wave = torchscience.signal_processing.waveform.sine_wave(
    torch.tensor(440.0),
    num_samples=48000,
    sample_rate=48000.0,
)
```

## Features

### Special Functions

| Function | Description |
|----------|-------------|
| `gamma(z)` | Gamma function with Lanczos approximation |
| `chebyshev_polynomial_t(n, x)` | Chebyshev polynomials of the first kind |
| `incomplete_beta(a, b, x)` | Regularized incomplete beta function |
| `hypergeometric_2_f_1(a, b, c, z)` | Gaussian hypergeometric function |

### Signal Processing

| Submodule | Function | Description |
|-----------|----------|-------------|
| `filter` | `butterworth_analog_bandpass_filter` | Analog Butterworth bandpass filter design |
| `waveform` | `sine_wave` | Sinusoidal waveform generation |
| `window_function` | `rectangular_window` | Rectangular (boxcar) window |

## PyTorch Compatibility

Every operator supports:

- **Autograd**: First-order and higher-order gradients, complex gradients (Wirtinger), forward-mode AD
- **Compilation**: `torch.compile()` for graph optimization and kernel fusion
- **Batching**: `torch.func.vmap` for efficient parallelism, NumPy-style broadcasting
- **Data types**: float16, bfloat16, float32, float64, complex64, complex128
- **Hardware**: CPU (vectorized), CUDA (device-agnostic kernels)
- **Layouts**: Strided, Sparse COO, Sparse CSR
- **Mixed precision**: Autocast with numerical stability

## Design Principles

1. **First-class PyTorch integration** — Seamless compatibility with autograd, torch.compile, torch.func, and the full PyTorch ecosystem.

2. **Numerical rigor** — Careful handling of edge cases, overflow/underflow, and precision across the full input range.

3. **Production quality** — Follows PyTorch standards for code organization, testing, and documentation.

## Documentation

- [Architecture](docs/architecture.rst) — Implementation patterns and PyTorch integration details
- [Roadmap](docs/roadmap.rst) — Planned modules and features

## Contributing

Contributions are welcome. See [Architecture](docs/architecture.rst) for implementation patterns.

1. Core implementation in `csrc/impl/` (device-agnostic, header-only C++)
2. Dispatcher registrations for CPU, CUDA, Autograd, Meta, Autocast, Sparse, Quantized
3. Python wrapper with comprehensive docstring
4. Tests using the mixin framework in `src/torchscience/testing/`
5. Benchmarks using Google Benchmark

## License

MIT License. See [LICENSE](LICENSE) for details.
