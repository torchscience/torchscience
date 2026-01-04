# Design: `torchscience.encryption` and `torchscience.privacy`

**Date:** 2026-01-04

**Status:** Approved

## Overview

Two new modules for cryptographic primitives and differential privacy mechanisms:

- `torchscience.encryption` — ChaCha20 stream cipher, SHA-256 hash, HMAC-SHA256
- `torchscience.privacy` — Gaussian, Laplace, Exponential, and Randomized Response mechanisms

## Module Structure

```
torchscience/
├── encryption/
│   ├── __init__.py
│   ├── _chacha20.py          # ChaCha20 stream cipher
│   ├── _sha256.py            # SHA-256 hash function
│   ├── _hmac.py              # HMAC-SHA256
│   └── _generator.py         # ChaCha20Generator class
└── privacy/
    ├── __init__.py
    ├── _gaussian.py          # Gaussian mechanism
    ├── _laplace.py           # Laplace mechanism
    ├── _exponential.py       # Exponential mechanism
    └── _randomized_response.py
```

## Operator Categories

| Operator | Category | Shape behavior |
|----------|----------|----------------|
| `chacha20` | Factory | `(n,) → (n,)` bytes as uint8 tensor |
| `sha256` | Fixed | `(..., n) → (..., 32)` bytes |
| `hmac_sha256` | Fixed | `(key, msg) → (..., 32)` bytes |
| `gaussian_mechanism` | Pointwise | Broadcasts noise to input shape |
| `laplace_mechanism` | Pointwise | Broadcasts noise to input shape |
| `exponential_mechanism` | Reduction | `(..., k) utilities → (...)` indices |
| `randomized_response` | Pointwise | Same shape as input |

## Encryption Module API

### ChaCha20

```python
def chacha20(
    key: Tensor,        # (32,) uint8 — 256-bit key
    nonce: Tensor,      # (12,) uint8 — 96-bit nonce
    counter: int,       # Block counter (default 0)
    num_bytes: int,     # Number of bytes to generate
    *,
    device: torch.device = None,
    dtype: torch.dtype = torch.uint8,
) -> Tensor:
    """Generate pseudorandom bytes using ChaCha20.

    Returns: (num_bytes,) uint8 tensor
    """
```

### SHA-256

```python
def sha256(
    data: Tensor,  # (..., n) uint8 — input bytes
) -> Tensor:
    """Compute SHA-256 hash.

    Returns: (..., 32) uint8 tensor — 256-bit hash
    """
```

### HMAC-SHA256

```python
def hmac_sha256(
    key: Tensor,   # (k,) uint8 — key of any length
    data: Tensor,  # (..., n) uint8 — message bytes
) -> Tensor:
    """Compute HMAC-SHA256.

    Returns: (..., 32) uint8 tensor — authentication tag
    """
```

### ChaCha20Generator

```python
class ChaCha20Generator:
    """Cryptographically secure RNG using ChaCha20.

    Compatible with torch.Generator interface patterns.
    """

    def __init__(
        self,
        seed: Tensor | int | None = None,  # 256-bit seed or int
        device: torch.device = None,
    ):
        ...

    def manual_seed(self, seed: int) -> "ChaCha20Generator":
        """Reset state with new seed."""

    def random_bytes(self, num_bytes: int) -> Tensor:
        """Generate raw random bytes as uint8."""

    def random(self, size: tuple, dtype: torch.dtype) -> Tensor:
        """Generate uniform random values in [0, 1)."""

    def randn(self, size: tuple, dtype: torch.dtype) -> Tensor:
        """Generate standard normal samples (Box-Muller)."""

    def get_state(self) -> Tensor:
        """Return internal state for checkpointing."""

    def set_state(self, state: Tensor) -> None:
        """Restore from checkpoint."""
```

## Privacy Module API

### Gaussian Mechanism

```python
def gaussian_mechanism(
    x: Tensor,              # Input tensor (any shape)
    sensitivity: float,     # L2 sensitivity of the query
    epsilon: float,         # Privacy parameter ε
    delta: float,           # Privacy parameter δ
    generator: Generator,   # ChaCha20-based CSPRNG
) -> Tensor:
    """Add Gaussian noise calibrated for (ε,δ)-differential privacy.

    Noise scale: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
    Returns: x + N(0, σ²)
    """
```

### Laplace Mechanism

```python
def laplace_mechanism(
    x: Tensor,              # Input tensor
    sensitivity: float,     # L1 sensitivity
    epsilon: float,         # Privacy parameter ε
    generator: Generator,
) -> Tensor:
    """Add Laplace noise calibrated for ε-differential privacy.

    Noise scale: b = sensitivity / ε
    Returns: x + Laplace(0, b)
    """
```

### Exponential Mechanism

```python
def exponential_mechanism(
    utilities: Tensor,      # (..., k) utility scores
    sensitivity: float,     # Sensitivity of utility function
    epsilon: float,
    generator: Generator,
) -> Tensor:
    """Sample index proportional to exp(ε * utility / (2 * sensitivity)).

    Returns: (...,) int64 tensor of selected indices
    """
```

### Randomized Response

```python
def randomized_response(
    x: Tensor,              # Boolean or integer tensor
    epsilon: float,
    generator: Generator,
    num_categories: int = 2,  # 2 for binary
) -> Tensor:
    """Flip each value with probability 1/(1 + exp(ε)).

    Returns: Tensor same shape/dtype as input
    """
```

## C++ Kernel Architecture

### Directory Structure

```
src/torchscience/csrc/
├── kernel/encryption/
│   ├── chacha20.h              # ChaCha20 quarter-round, block function
│   ├── sha256.h                # SHA-256 compression function
│   └── hmac.h                  # HMAC construction (uses sha256)
├── kernel/privacy/
│   ├── gaussian_mechanism.h
│   ├── gaussian_mechanism_backward.h
│   ├── laplace_mechanism.h
│   ├── laplace_mechanism_backward.h
│   ├── exponential_mechanism.h        # No backward (discrete output)
│   └── randomized_response.h          # No backward (discrete output)
├── cpu/encryption/
│   ├── chacha20.h
│   ├── sha256.h
│   └── hmac.h
├── cpu/privacy/
│   └── mechanisms.h
├── cuda/encryption/
│   ├── chacha20.cu             # Parallel block generation
│   ├── sha256.cu               # Parallel hashing
│   └── hmac.cu
├── meta/encryption/
│   ├── chacha20.h
│   ├── sha256.h
│   └── hmac.h
├── meta/privacy/
│   └── mechanisms.h
└── autograd/privacy/
    └── mechanisms.h
```

### Implementation Strategy

| Operator | CPU Strategy | CUDA Strategy |
|----------|--------------|---------------|
| `chacha20` | Vectorized quarter-rounds, 64-byte blocks | One thread per block, parallel across blocks |
| `sha256` | 64-byte chunks sequentially per hash | One thread per hash, parallel across batch |
| `hmac` | Two SHA-256 calls (inner/outer) | Same parallelism as sha256 |
| `gaussian_mechanism` | Pointwise: x + σ * noise | Parallel across elements |
| `laplace_mechanism` | Pointwise: x + b * noise | Parallel across elements |

### Autograd Behavior

Privacy mechanisms support gradients through the input `x`:

```cpp
// Forward: y = x + noise
// Backward: dx = dy (noise is treated as constant)
```

The noise tensor is generated by `ChaCha20Generator` in Python and passed to the C++ kernel. The kernel detaches the noise to prevent gradients flowing through the sampling process.

**No autograd** for encryption primitives (non-differentiable discrete operations).

## Testing Strategy

### Encryption Tests

| Test | Verification |
|------|--------------|
| `test_chacha20_rfc8439_vectors` | Output matches RFC 8439 test vectors |
| `test_sha256_nist_vectors` | Output matches NIST CAVP test vectors |
| `test_hmac_rfc4231_vectors` | Output matches RFC 4231 test vectors |
| `test_chacha20_determinism` | Same key/nonce/counter → same output |
| `test_sha256_batched` | Batched hashing correctness |
| `test_cuda_cpu_parity` | CUDA and CPU produce identical results |
| `test_meta_shapes` | Meta tensors infer correct output shapes |

### Privacy Tests

| Test | Verification |
|------|--------------|
| `test_gaussian_noise_scale` | Empirical σ matches theoretical σ |
| `test_laplace_noise_scale` | Empirical b matches theoretical b |
| `test_exponential_selection_distribution` | Selection probabilities match theory |
| `test_randomized_response_flip_rate` | Flip probability matches 1/(1+exp(ε)) |
| `test_gradcheck` | Gradients correct for gaussian/laplace |
| `test_gradgradcheck` | Second-order gradients verified |
| `test_determinism_with_generator` | Same generator state → same noise |
| `test_generator_state_save_restore` | Checkpointing works correctly |

## Implementation Order

### Phase 1: Foundation (encryption module)

1. `chacha20` — Core CSPRNG primitive
2. `ChaCha20Generator` — Python wrapper with `randn()`, `random()`
3. `sha256` — Hash function
4. `hmac_sha256` — Builds on sha256

### Phase 2: Privacy mechanisms

5. `gaussian_mechanism` — Most common, uses generator.randn()
6. `laplace_mechanism` — Uses generator for Laplace samples
7. `exponential_mechanism` — Discrete selection
8. `randomized_response` — Binary/categorical flipping

### Dependency Graph

```
chacha20 → ChaCha20Generator → gaussian_mechanism
                             → laplace_mechanism
                             → exponential_mechanism
                             → randomized_response

sha256 → hmac_sha256
```

## Scope Estimate

| Component | Files | Complexity |
|-----------|-------|------------|
| ChaCha20 (kernel + CPU + CUDA + meta) | 4 | Medium |
| SHA-256 (kernel + CPU + CUDA + meta) | 4 | Medium |
| HMAC | 2 | Low |
| ChaCha20Generator | 1 | Low |
| Privacy mechanisms (4 ops × 4 backends) | 8 | Low-Medium |
| Tests | 2 | Medium |