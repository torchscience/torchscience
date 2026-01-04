# Encryption and Privacy Modules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.encryption` (ChaCha20, SHA-256, HMAC) and `torchscience.privacy` (Gaussian, Laplace, Exponential, Randomized Response mechanisms).

**Architecture:** Two modules with C++ kernels for performance. Encryption primitives operate on uint8 tensors (raw bytes). Privacy mechanisms use the ChaCha20Generator for cryptographically secure randomness and support autograd through the input tensor.

**Tech Stack:** C++17 kernels, PyTorch dispatcher, TORCH_LIBRARY registration, CPU/CUDA/Meta backends.

---

## Phase 1: ChaCha20 Stream Cipher

### Task 1: Create Module Structure

**Files:**
- Create: `src/torchscience/encryption/__init__.py`
- Create: `src/torchscience/encryption/_chacha20.py`
- Create: `tests/torchscience/encryption/__init__.py`
- Create: `tests/torchscience/encryption/test__chacha20.py`

**Step 1: Create encryption module directory and __init__.py**

```python
# src/torchscience/encryption/__init__.py
from torchscience.encryption._chacha20 import chacha20

__all__ = [
    "chacha20",
]
```

**Step 2: Create stub Python wrapper**

```python
# src/torchscience/encryption/_chacha20.py
import torch
from torch import Tensor


def chacha20(
    key: Tensor,
    nonce: Tensor,
    num_bytes: int,
    counter: int = 0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.uint8,
) -> Tensor:
    """Generate pseudorandom bytes using ChaCha20 stream cipher.

    Implements RFC 8439 ChaCha20.

    Parameters
    ----------
    key : Tensor
        256-bit key as (32,) uint8 tensor.
    nonce : Tensor
        96-bit nonce as (12,) uint8 tensor.
    num_bytes : int
        Number of bytes to generate.
    counter : int, optional
        Initial block counter, default 0.
    device : torch.device, optional
        Output device. Defaults to key's device.
    dtype : torch.dtype, optional
        Output dtype, default torch.uint8.

    Returns
    -------
    Tensor
        (num_bytes,) tensor of pseudorandom bytes.
    """
    return torch.ops.torchscience.chacha20(key, nonce, num_bytes, counter)
```

**Step 3: Create test file with RFC 8439 test vector**

```python
# tests/torchscience/encryption/__init__.py
```

```python
# tests/torchscience/encryption/test__chacha20.py
import pytest
import torch

from torchscience.encryption import chacha20


class TestChaCha20:
    """Tests for ChaCha20 stream cipher."""

    def test_rfc8439_test_vector(self):
        """Test against RFC 8439 Section 2.4.2 test vector."""
        # Key: 00:01:02:...:1f
        key = torch.arange(32, dtype=torch.uint8)

        # Nonce: 00:00:00:09:00:00:00:4a:00:00:00:00
        nonce = torch.tensor(
            [0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4A, 0x00, 0x00, 0x00, 0x00],
            dtype=torch.uint8,
        )

        # Counter = 1
        output = chacha20(key, nonce, num_bytes=64, counter=1)

        # Expected first 16 bytes from RFC 8439
        expected_start = torch.tensor(
            [0x10, 0xF1, 0xE7, 0xE4, 0xD1, 0x3B, 0x59, 0x15,
             0x50, 0x0F, 0xDD, 0x1F, 0xA3, 0x20, 0x71, 0xC4],
            dtype=torch.uint8,
        )

        assert output.shape == (64,)
        assert output.dtype == torch.uint8
        torch.testing.assert_close(output[:16], expected_start)

    def test_key_shape_validation(self):
        """Key must be exactly 32 bytes."""
        key = torch.zeros(16, dtype=torch.uint8)  # Wrong size
        nonce = torch.zeros(12, dtype=torch.uint8)

        with pytest.raises(RuntimeError, match="key"):
            chacha20(key, nonce, num_bytes=64)

    def test_nonce_shape_validation(self):
        """Nonce must be exactly 12 bytes."""
        key = torch.zeros(32, dtype=torch.uint8)
        nonce = torch.zeros(8, dtype=torch.uint8)  # Wrong size

        with pytest.raises(RuntimeError, match="nonce"):
            chacha20(key, nonce, num_bytes=64)

    def test_determinism(self):
        """Same inputs produce same outputs."""
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.zeros(12, dtype=torch.uint8)

        out1 = chacha20(key, nonce, num_bytes=128)
        out2 = chacha20(key, nonce, num_bytes=128)

        torch.testing.assert_close(out1, out2)

    def test_different_counters_different_output(self):
        """Different counters produce different blocks."""
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.zeros(12, dtype=torch.uint8)

        out0 = chacha20(key, nonce, num_bytes=64, counter=0)
        out1 = chacha20(key, nonce, num_bytes=64, counter=1)

        assert not torch.equal(out0, out1)

    def test_output_length(self):
        """Output has requested length."""
        key = torch.arange(32, dtype=torch.uint8)
        nonce = torch.zeros(12, dtype=torch.uint8)

        for length in [1, 63, 64, 65, 128, 1000]:
            out = chacha20(key, nonce, num_bytes=length)
            assert out.shape == (length,)

    def test_meta_tensor(self):
        """Meta tensors produce correct shape without computation."""
        key = torch.zeros(32, dtype=torch.uint8, device="meta")
        nonce = torch.zeros(12, dtype=torch.uint8, device="meta")

        out = chacha20(key, nonce, num_bytes=256)

        assert out.device.type == "meta"
        assert out.shape == (256,)
        assert out.dtype == torch.uint8
```

**Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__chacha20.py -v`
Expected: FAIL (operator not registered)

**Step 5: Commit module structure**

```bash
git add src/torchscience/encryption/ tests/torchscience/encryption/
git commit -m "test(encryption): add failing tests for chacha20"
```

---

### Task 2: Register ChaCha20 Schema

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add chacha20 schema definition**

Find the TORCH_LIBRARY block and add:

```cpp
// Encryption operators
m.def("chacha20(Tensor key, Tensor nonce, int num_bytes, int counter=0) -> Tensor");
```

**Step 2: Run tests to verify schema registered**

Run: `.venv/bin/python -c "import torch; torch.ops.torchscience.chacha20"`
Expected: Returns function object (not AttributeError)

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(encryption): add chacha20 schema definition"
```

---

### Task 3: Implement ChaCha20 Kernel

**Files:**
- Create: `src/torchscience/csrc/kernel/encryption/chacha20.h`

**Step 1: Write header-only kernel implementation**

```cpp
// src/torchscience/csrc/kernel/encryption/chacha20.h
#pragma once

#include <cstdint>
#include <array>

namespace torchscience::kernel::encryption {

// ChaCha20 quarter round
inline void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// ChaCha20 block function
inline void chacha20_block(
    std::array<uint32_t, 16>& state,
    const std::array<uint32_t, 16>& input
) {
    state = input;

    // 20 rounds (10 double rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarter_round(state[0], state[4], state[8],  state[12]);
        quarter_round(state[1], state[5], state[9],  state[13]);
        quarter_round(state[2], state[6], state[10], state[14]);
        quarter_round(state[3], state[7], state[11], state[15]);
        // Diagonal rounds
        quarter_round(state[0], state[5], state[10], state[15]);
        quarter_round(state[1], state[6], state[11], state[12]);
        quarter_round(state[2], state[7], state[8],  state[13]);
        quarter_round(state[3], state[4], state[9],  state[14]);
    }

    // Add input to state
    for (int i = 0; i < 16; i++) {
        state[i] += input[i];
    }
}

// Load 32-bit little-endian word from bytes
inline uint32_t load_le32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0])
         | (static_cast<uint32_t>(bytes[1]) << 8)
         | (static_cast<uint32_t>(bytes[2]) << 16)
         | (static_cast<uint32_t>(bytes[3]) << 24);
}

// Store 32-bit word as little-endian bytes
inline void store_le32(uint8_t* bytes, uint32_t word) {
    bytes[0] = static_cast<uint8_t>(word);
    bytes[1] = static_cast<uint8_t>(word >> 8);
    bytes[2] = static_cast<uint8_t>(word >> 16);
    bytes[3] = static_cast<uint8_t>(word >> 24);
}

// Initialize ChaCha20 state from key, nonce, counter
inline void chacha20_init(
    std::array<uint32_t, 16>& state,
    const uint8_t* key,
    const uint8_t* nonce,
    uint32_t counter
) {
    // Constants: "expand 32-byte k"
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

    // Key (8 words)
    for (int i = 0; i < 8; i++) {
        state[4 + i] = load_le32(key + 4 * i);
    }

    // Counter
    state[12] = counter;

    // Nonce (3 words)
    for (int i = 0; i < 3; i++) {
        state[13 + i] = load_le32(nonce + 4 * i);
    }
}

// Generate keystream bytes
inline void chacha20_keystream(
    uint8_t* output,
    int64_t num_bytes,
    const uint8_t* key,
    const uint8_t* nonce,
    uint32_t counter
) {
    std::array<uint32_t, 16> input_state;
    std::array<uint32_t, 16> output_state;

    chacha20_init(input_state, key, nonce, counter);

    int64_t offset = 0;
    while (offset < num_bytes) {
        chacha20_block(output_state, input_state);

        // Copy output bytes
        int64_t block_bytes = std::min(static_cast<int64_t>(64), num_bytes - offset);
        for (int64_t i = 0; i < block_bytes; i++) {
            output[offset + i] = reinterpret_cast<uint8_t*>(output_state.data())[i];
        }

        offset += 64;
        input_state[12]++;  // Increment counter
    }
}

}  // namespace torchscience::kernel::encryption
```

**Step 2: Commit kernel**

```bash
git add src/torchscience/csrc/kernel/encryption/chacha20.h
git commit -m "feat(encryption): implement chacha20 kernel"
```

---

### Task 4: Implement ChaCha20 CPU Backend

**Files:**
- Create: `src/torchscience/csrc/cpu/encryption/chacha20.h`

**Step 1: Write CPU backend**

```cpp
// src/torchscience/csrc/cpu/encryption/chacha20.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../../kernel/encryption/chacha20.h"

namespace torchscience::cpu::encryption {

at::Tensor chacha20(
    const at::Tensor& key,
    const at::Tensor& nonce,
    int64_t num_bytes,
    int64_t counter
) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20: key must be a 1D tensor of 32 bytes, got shape ", key.sizes());
    TORCH_CHECK(key.dtype() == at::kByte,
        "chacha20: key must be uint8, got ", key.dtype());
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20: nonce must be a 1D tensor of 12 bytes, got shape ", nonce.sizes());
    TORCH_CHECK(nonce.dtype() == at::kByte,
        "chacha20: nonce must be uint8, got ", nonce.dtype());
    TORCH_CHECK(num_bytes > 0,
        "chacha20: num_bytes must be positive, got ", num_bytes);
    TORCH_CHECK(counter >= 0,
        "chacha20: counter must be non-negative, got ", counter);

    auto key_contig = key.contiguous();
    auto nonce_contig = nonce.contiguous();

    auto output = at::empty({num_bytes}, key.options().dtype(at::kByte));

    kernel::encryption::chacha20_keystream(
        output.data_ptr<uint8_t>(),
        num_bytes,
        key_contig.data_ptr<uint8_t>(),
        nonce_contig.data_ptr<uint8_t>(),
        static_cast<uint32_t>(counter)
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("chacha20", &chacha20);
}

}  // namespace torchscience::cpu::encryption
```

**Step 2: Include in main compilation unit**

Add to `src/torchscience/csrc/torchscience.cpp` after existing includes:

```cpp
#include "cpu/encryption/chacha20.h"
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__chacha20.py -v -k "not meta"`
Expected: All non-meta tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/encryption/chacha20.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(encryption): implement chacha20 CPU backend"
```

---

### Task 5: Implement ChaCha20 Meta Backend

**Files:**
- Create: `src/torchscience/csrc/meta/encryption/chacha20.h`

**Step 1: Write meta backend**

```cpp
// src/torchscience/csrc/meta/encryption/chacha20.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor chacha20(
    const at::Tensor& key,
    const at::Tensor& nonce,
    int64_t num_bytes,
    int64_t counter
) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20: key must be a 1D tensor of 32 bytes");
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20: nonce must be a 1D tensor of 12 bytes");
    TORCH_CHECK(num_bytes > 0,
        "chacha20: num_bytes must be positive");

    return at::empty({num_bytes}, key.options().dtype(at::kByte));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("chacha20", &chacha20);
}

}  // namespace torchscience::meta::encryption
```

**Step 2: Include in main compilation unit**

Add to `src/torchscience/csrc/torchscience.cpp`:

```cpp
#include "meta/encryption/chacha20.h"
```

**Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__chacha20.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/torchscience/csrc/meta/encryption/chacha20.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(encryption): implement chacha20 meta backend"
```

---

## Phase 2: ChaCha20Generator

### Task 6: Implement ChaCha20Generator

**Files:**
- Create: `src/torchscience/encryption/_generator.py`
- Modify: `src/torchscience/encryption/__init__.py`
- Create: `tests/torchscience/encryption/test__generator.py`

**Step 1: Write failing tests**

```python
# tests/torchscience/encryption/test__generator.py
import pytest
import torch
import math

from torchscience.encryption import ChaCha20Generator


class TestChaCha20Generator:
    """Tests for ChaCha20Generator CSPRNG."""

    def test_construction_with_int_seed(self):
        """Can construct with integer seed."""
        gen = ChaCha20Generator(seed=42)
        assert gen is not None

    def test_construction_with_tensor_seed(self):
        """Can construct with 32-byte tensor seed."""
        seed = torch.arange(32, dtype=torch.uint8)
        gen = ChaCha20Generator(seed=seed)
        assert gen is not None

    def test_random_bytes_length(self):
        """random_bytes returns correct length."""
        gen = ChaCha20Generator(seed=42)
        for length in [1, 32, 64, 100, 1000]:
            out = gen.random_bytes(length)
            assert out.shape == (length,)
            assert out.dtype == torch.uint8

    def test_random_shape(self):
        """random returns correct shape and range."""
        gen = ChaCha20Generator(seed=42)
        out = gen.random((100, 100), dtype=torch.float32)

        assert out.shape == (100, 100)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0
        assert out.max() < 1.0

    def test_randn_shape(self):
        """randn returns correct shape."""
        gen = ChaCha20Generator(seed=42)
        out = gen.randn((100, 100), dtype=torch.float32)

        assert out.shape == (100, 100)
        assert out.dtype == torch.float32

    def test_randn_distribution(self):
        """randn produces approximately standard normal."""
        gen = ChaCha20Generator(seed=42)
        out = gen.randn((10000,), dtype=torch.float64)

        # Mean should be close to 0
        assert abs(out.mean().item()) < 0.1
        # Std should be close to 1
        assert abs(out.std().item() - 1.0) < 0.1

    def test_determinism(self):
        """Same seed produces same sequence."""
        gen1 = ChaCha20Generator(seed=123)
        gen2 = ChaCha20Generator(seed=123)

        out1 = gen1.random_bytes(100)
        out2 = gen2.random_bytes(100)

        torch.testing.assert_close(out1, out2)

    def test_different_seeds_different_output(self):
        """Different seeds produce different sequences."""
        gen1 = ChaCha20Generator(seed=1)
        gen2 = ChaCha20Generator(seed=2)

        out1 = gen1.random_bytes(100)
        out2 = gen2.random_bytes(100)

        assert not torch.equal(out1, out2)

    def test_manual_seed(self):
        """manual_seed resets state."""
        gen = ChaCha20Generator(seed=42)
        out1 = gen.random_bytes(100)

        gen.manual_seed(42)
        out2 = gen.random_bytes(100)

        torch.testing.assert_close(out1, out2)

    def test_get_set_state(self):
        """State can be saved and restored."""
        gen = ChaCha20Generator(seed=42)
        gen.random_bytes(50)  # Advance state

        state = gen.get_state()
        out1 = gen.random_bytes(100)

        gen.set_state(state)
        out2 = gen.random_bytes(100)

        torch.testing.assert_close(out1, out2)

    def test_device_parameter(self):
        """Generator respects device parameter."""
        gen = ChaCha20Generator(seed=42, device="cpu")
        out = gen.random_bytes(64)
        assert out.device.type == "cpu"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__generator.py -v`
Expected: FAIL (ChaCha20Generator not defined)

**Step 3: Implement ChaCha20Generator**

```python
# src/torchscience/encryption/_generator.py
import hashlib
import math
from typing import Tuple

import torch
from torch import Tensor

from torchscience.encryption._chacha20 import chacha20


class ChaCha20Generator:
    """Cryptographically secure random number generator using ChaCha20.

    This generator provides a PyTorch-compatible interface for generating
    random numbers using the ChaCha20 stream cipher as the underlying CSPRNG.

    Parameters
    ----------
    seed : int | Tensor | None, optional
        Seed for the generator. Can be:
        - int: Hashed to 256 bits using SHA-256
        - Tensor: Must be (32,) uint8 for direct use as key
        - None: Uses random seed from system entropy
    device : torch.device | None, optional
        Device for generated tensors. Default is CPU.

    Examples
    --------
    >>> gen = ChaCha20Generator(seed=42)
    >>> gen.random((3, 3), dtype=torch.float32)
    tensor([[0.1234, 0.5678, ...], ...])
    >>> gen.randn((1000,), dtype=torch.float64)
    tensor([0.1234, -0.5678, ...])
    """

    def __init__(
        self,
        seed: int | Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._device = torch.device(device if device is not None else "cpu")
        self._counter = 0
        self._nonce = torch.zeros(12, dtype=torch.uint8, device=self._device)

        if seed is None:
            # Use system entropy
            self._key = torch.randint(
                0, 256, (32,), dtype=torch.uint8, device=self._device
            )
        elif isinstance(seed, int):
            self.manual_seed(seed)
        elif isinstance(seed, Tensor):
            if seed.shape != (32,) or seed.dtype != torch.uint8:
                raise ValueError("Tensor seed must be (32,) uint8")
            self._key = seed.to(device=self._device)
        else:
            raise TypeError(f"seed must be int, Tensor, or None, got {type(seed)}")

    def manual_seed(self, seed: int) -> "ChaCha20Generator":
        """Reset generator with a new integer seed.

        Parameters
        ----------
        seed : int
            Integer seed (hashed to 256 bits).

        Returns
        -------
        ChaCha20Generator
            Self for method chaining.
        """
        # Hash integer to 256 bits
        seed_bytes = seed.to_bytes(
            (seed.bit_length() + 7) // 8 or 1,
            byteorder="little",
            signed=(seed < 0),
        )
        key_bytes = hashlib.sha256(seed_bytes).digest()
        self._key = torch.tensor(
            list(key_bytes), dtype=torch.uint8, device=self._device
        )
        self._counter = 0
        return self

    def random_bytes(self, num_bytes: int) -> Tensor:
        """Generate random bytes.

        Parameters
        ----------
        num_bytes : int
            Number of bytes to generate.

        Returns
        -------
        Tensor
            (num_bytes,) uint8 tensor of random bytes.
        """
        output = chacha20(
            self._key, self._nonce, num_bytes=num_bytes, counter=self._counter
        )
        # Update counter: each block is 64 bytes
        self._counter += (num_bytes + 63) // 64
        return output

    def random(
        self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Generate uniform random values in [0, 1).

        Parameters
        ----------
        size : tuple of int
            Output shape.
        dtype : torch.dtype, optional
            Output dtype (float16, float32, or float64).

        Returns
        -------
        Tensor
            Tensor of uniform random values.
        """
        numel = math.prod(size)

        if dtype == torch.float64:
            # Need 8 bytes per value (use 53 bits for mantissa)
            bytes_needed = numel * 8
            raw = self.random_bytes(bytes_needed)
            # View as uint64, mask to 53 bits, convert to float64
            raw_64 = raw.view(torch.int64)
            raw_64 = raw_64 & ((1 << 53) - 1)
            result = raw_64.to(torch.float64) / (1 << 53)
        elif dtype == torch.float32:
            # Need 4 bytes per value (use 24 bits for mantissa)
            bytes_needed = numel * 4
            raw = self.random_bytes(bytes_needed)
            raw_32 = raw.view(torch.int32)
            raw_32 = raw_32 & ((1 << 24) - 1)
            result = raw_32.to(torch.float32) / (1 << 24)
        elif dtype == torch.float16:
            # Need 2 bytes per value (use 11 bits for mantissa)
            bytes_needed = numel * 2
            raw = self.random_bytes(bytes_needed)
            raw_16 = raw.view(torch.int16)
            raw_16 = raw_16 & ((1 << 11) - 1)
            result = raw_16.to(torch.float16) / (1 << 11)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return result.view(size)

    def randn(
        self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Generate standard normal random values using Box-Muller transform.

        Parameters
        ----------
        size : tuple of int
            Output shape.
        dtype : torch.dtype, optional
            Output dtype.

        Returns
        -------
        Tensor
            Tensor of standard normal random values.
        """
        numel = math.prod(size)
        # Box-Muller needs pairs
        n_pairs = (numel + 1) // 2

        # Generate uniform pairs
        u1 = self.random((n_pairs,), dtype=dtype)
        u2 = self.random((n_pairs,), dtype=dtype)

        # Avoid log(0)
        u1 = u1.clamp(min=1e-10)

        # Box-Muller transform
        r = torch.sqrt(-2 * torch.log(u1))
        theta = 2 * math.pi * u2
        z0 = r * torch.cos(theta)
        z1 = r * torch.sin(theta)

        # Interleave and truncate to desired size
        result = torch.stack([z0, z1], dim=-1).flatten()[:numel]
        return result.view(size)

    def get_state(self) -> Tensor:
        """Get generator state for checkpointing.

        Returns
        -------
        Tensor
            State tensor (key + counter).
        """
        counter_tensor = torch.tensor(
            [self._counter], dtype=torch.int64, device=self._device
        )
        return torch.cat([self._key.to(torch.int64), counter_tensor])

    def set_state(self, state: Tensor) -> None:
        """Restore generator state from checkpoint.

        Parameters
        ----------
        state : Tensor
            State tensor from get_state().
        """
        self._key = state[:32].to(torch.uint8)
        self._counter = state[32].item()
```

**Step 4: Update __init__.py**

```python
# src/torchscience/encryption/__init__.py
from torchscience.encryption._chacha20 import chacha20
from torchscience.encryption._generator import ChaCha20Generator

__all__ = [
    "chacha20",
    "ChaCha20Generator",
]
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__generator.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/torchscience/encryption/_generator.py src/torchscience/encryption/__init__.py tests/torchscience/encryption/test__generator.py
git commit -m "feat(encryption): implement ChaCha20Generator"
```

---

## Phase 3: SHA-256

### Task 7: Implement SHA-256 Kernel

**Files:**
- Create: `src/torchscience/csrc/kernel/encryption/sha256.h`
- Create: `src/torchscience/encryption/_sha256.py`
- Create: `tests/torchscience/encryption/test__sha256.py`

**Step 1: Write failing tests**

```python
# tests/torchscience/encryption/test__sha256.py
import pytest
import torch
import hashlib

from torchscience.encryption import sha256


class TestSHA256:
    """Tests for SHA-256 hash function."""

    def test_empty_input(self):
        """Hash of empty input matches known value."""
        data = torch.tensor([], dtype=torch.uint8)
        result = sha256(data)

        # SHA-256 of empty string
        expected = hashlib.sha256(b"").digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        assert result.shape == (32,)
        torch.testing.assert_close(result, expected_tensor)

    def test_abc_input(self):
        """Hash of 'abc' matches NIST test vector."""
        data = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)  # "abc"
        result = sha256(data)

        expected = hashlib.sha256(b"abc").digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_448_bit_input(self):
        """Hash of 448-bit input (56 bytes) matches hashlib."""
        data = torch.tensor(list(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), dtype=torch.uint8)
        result = sha256(data)

        expected = hashlib.sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_various_lengths(self):
        """Hash of various input lengths matches hashlib."""
        for length in [1, 55, 56, 63, 64, 65, 100, 1000]:
            data_bytes = bytes(range(256)) * (length // 256 + 1)
            data_bytes = data_bytes[:length]
            data = torch.tensor(list(data_bytes), dtype=torch.uint8)

            result = sha256(data)
            expected = hashlib.sha256(data_bytes).digest()
            expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

            torch.testing.assert_close(result, expected_tensor)

    def test_batched(self):
        """Batched hashing produces correct per-element hashes."""
        # Two messages
        msg1 = b"hello"
        msg2 = b"world"

        # Pad to same length
        max_len = max(len(msg1), len(msg2))
        data = torch.zeros((2, max_len), dtype=torch.uint8)
        data[0, :len(msg1)] = torch.tensor(list(msg1), dtype=torch.uint8)
        data[1, :len(msg2)] = torch.tensor(list(msg2), dtype=torch.uint8)

        result = sha256(data)

        assert result.shape == (2, 32)
        torch.testing.assert_close(
            result[0],
            torch.tensor(list(hashlib.sha256(msg1).digest()), dtype=torch.uint8)
        )

    def test_determinism(self):
        """Same input produces same hash."""
        data = torch.arange(100, dtype=torch.uint8)
        result1 = sha256(data)
        result2 = sha256(data)

        torch.testing.assert_close(result1, result2)

    def test_meta_tensor(self):
        """Meta tensor produces correct shape."""
        data = torch.zeros(100, dtype=torch.uint8, device="meta")
        result = sha256(data)

        assert result.device.type == "meta"
        assert result.shape == (32,)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__sha256.py -v`
Expected: FAIL (sha256 not found)

**Step 3: Write Python wrapper**

```python
# src/torchscience/encryption/_sha256.py
import torch
from torch import Tensor


def sha256(data: Tensor) -> Tensor:
    """Compute SHA-256 hash.

    Parameters
    ----------
    data : Tensor
        Input bytes as (..., n) uint8 tensor.

    Returns
    -------
    Tensor
        (..., 32) uint8 tensor containing the 256-bit hash.
    """
    return torch.ops.torchscience.sha256(data)
```

**Step 4: Update __init__.py**

Add to `src/torchscience/encryption/__init__.py`:

```python
from torchscience.encryption._sha256 import sha256
# Add to __all__
```

**Step 5: Commit tests and stubs**

```bash
git add src/torchscience/encryption/_sha256.py src/torchscience/encryption/__init__.py tests/torchscience/encryption/test__sha256.py
git commit -m "test(encryption): add failing tests for sha256"
```

---

### Task 8: Implement SHA-256 Kernel and Backends

**Files:**
- Create: `src/torchscience/csrc/kernel/encryption/sha256.h`
- Create: `src/torchscience/csrc/cpu/encryption/sha256.h`
- Create: `src/torchscience/csrc/meta/encryption/sha256.h`
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add schema**

Add to `src/torchscience/csrc/torchscience.cpp`:

```cpp
m.def("sha256(Tensor data) -> Tensor");
```

**Step 2: Write kernel**

```cpp
// src/torchscience/csrc/kernel/encryption/sha256.h
#pragma once

#include <cstdint>
#include <array>
#include <cstring>

namespace torchscience::kernel::encryption {

// SHA-256 constants
constexpr std::array<uint32_t, 64> SHA256_K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

inline void sha256_transform(std::array<uint32_t, 8>& state, const uint8_t* block) {
    std::array<uint32_t, 64> w;

    // Load block into first 16 words (big-endian)
    for (int i = 0; i < 16; i++) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24)
             | (static_cast<uint32_t>(block[i * 4 + 1]) << 16)
             | (static_cast<uint32_t>(block[i * 4 + 2]) << 8)
             | static_cast<uint32_t>(block[i * 4 + 3]);
    }

    // Extend to 64 words
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + SHA256_K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e;
        e = d + t1;
        d = c; c = b; b = a;
        a = t1 + t2;
    }

    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

inline void sha256_hash(
    uint8_t* output,
    const uint8_t* input,
    int64_t input_len
) {
    // Initial hash values
    std::array<uint32_t, 8> state = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Process complete blocks
    int64_t num_blocks = input_len / 64;
    for (int64_t i = 0; i < num_blocks; i++) {
        sha256_transform(state, input + i * 64);
    }

    // Pad final block(s)
    uint8_t final_blocks[128];
    std::memset(final_blocks, 0, 128);

    int64_t remaining = input_len % 64;
    std::memcpy(final_blocks, input + num_blocks * 64, remaining);

    // Append 1 bit
    final_blocks[remaining] = 0x80;

    // Length in bits (big-endian, 64-bit)
    uint64_t bit_len = input_len * 8;
    int pad_blocks = (remaining < 56) ? 1 : 2;

    int len_offset = pad_blocks * 64 - 8;
    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = static_cast<uint8_t>(bit_len >> (56 - i * 8));
    }

    // Process padded blocks
    for (int i = 0; i < pad_blocks; i++) {
        sha256_transform(state, final_blocks + i * 64);
    }

    // Write output (big-endian)
    for (int i = 0; i < 8; i++) {
        output[i * 4]     = static_cast<uint8_t>(state[i] >> 24);
        output[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
        output[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
        output[i * 4 + 3] = static_cast<uint8_t>(state[i]);
    }
}

}  // namespace torchscience::kernel::encryption
```

**Step 3: Write CPU backend**

```cpp
// src/torchscience/csrc/cpu/encryption/sha256.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../../kernel/encryption/sha256.h"

namespace torchscience::cpu::encryption {

at::Tensor sha256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte,
        "sha256: data must be uint8, got ", data.dtype());

    auto data_contig = data.contiguous();

    if (data.dim() == 1) {
        // Single hash
        auto output = at::empty({32}, data.options());
        kernel::encryption::sha256_hash(
            output.data_ptr<uint8_t>(),
            data_contig.data_ptr<uint8_t>(),
            data.size(0)
        );
        return output;
    } else {
        // Batched: hash along last dimension
        auto batch_sizes = data.sizes().vec();
        int64_t msg_len = batch_sizes.back();
        batch_sizes.back() = 32;

        auto output = at::empty(batch_sizes, data.options());
        int64_t batch_size = data.numel() / msg_len;

        auto data_ptr = data_contig.data_ptr<uint8_t>();
        auto output_ptr = output.data_ptr<uint8_t>();

        for (int64_t i = 0; i < batch_size; i++) {
            kernel::encryption::sha256_hash(
                output_ptr + i * 32,
                data_ptr + i * msg_len,
                msg_len
            );
        }
        return output;
    }
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("sha256", &sha256);
}

}  // namespace torchscience::cpu::encryption
```

**Step 4: Write meta backend**

```cpp
// src/torchscience/csrc/meta/encryption/sha256.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor sha256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte,
        "sha256: data must be uint8");

    if (data.dim() == 1) {
        return at::empty({32}, data.options());
    } else {
        auto sizes = data.sizes().vec();
        sizes.back() = 32;
        return at::empty(sizes, data.options());
    }
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("sha256", &sha256);
}

}  // namespace torchscience::meta::encryption
```

**Step 5: Include in torchscience.cpp**

```cpp
#include "cpu/encryption/sha256.h"
#include "meta/encryption/sha256.h"
```

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__sha256.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/torchscience/csrc/kernel/encryption/sha256.h src/torchscience/csrc/cpu/encryption/sha256.h src/torchscience/csrc/meta/encryption/sha256.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(encryption): implement sha256"
```

---

## Phase 4: HMAC-SHA256

### Task 9: Implement HMAC-SHA256

**Files:**
- Create: `src/torchscience/encryption/_hmac.py`
- Create: `tests/torchscience/encryption/test__hmac.py`

**Step 1: Write failing tests**

```python
# tests/torchscience/encryption/test__hmac.py
import pytest
import torch
import hmac as py_hmac
import hashlib

from torchscience.encryption import hmac_sha256


class TestHMAC:
    """Tests for HMAC-SHA256."""

    def test_rfc4231_test_case_1(self):
        """RFC 4231 test case 1."""
        key = torch.tensor([0x0b] * 20, dtype=torch.uint8)
        data = torch.tensor(list(b"Hi There"), dtype=torch.uint8)

        result = hmac_sha256(key, data)

        expected = py_hmac.new(bytes([0x0b] * 20), b"Hi There", hashlib.sha256).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_rfc4231_test_case_2(self):
        """RFC 4231 test case 2: key = 'Jefe'."""
        key = torch.tensor(list(b"Jefe"), dtype=torch.uint8)
        data = torch.tensor(list(b"what do ya want for nothing?"), dtype=torch.uint8)

        result = hmac_sha256(key, data)

        expected = py_hmac.new(b"Jefe", b"what do ya want for nothing?", hashlib.sha256).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_long_key(self):
        """Key longer than block size is hashed."""
        key = torch.arange(100, dtype=torch.uint8)  # > 64 bytes
        data = torch.tensor(list(b"test message"), dtype=torch.uint8)

        result = hmac_sha256(key, data)

        expected = py_hmac.new(bytes(range(100)), b"test message", hashlib.sha256).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_empty_message(self):
        """HMAC of empty message."""
        key = torch.tensor(list(b"key"), dtype=torch.uint8)
        data = torch.tensor([], dtype=torch.uint8)

        result = hmac_sha256(key, data)

        expected = py_hmac.new(b"key", b"", hashlib.sha256).digest()
        expected_tensor = torch.tensor(list(expected), dtype=torch.uint8)

        torch.testing.assert_close(result, expected_tensor)

    def test_output_shape(self):
        """Output is always 32 bytes."""
        key = torch.zeros(16, dtype=torch.uint8)
        data = torch.zeros(100, dtype=torch.uint8)

        result = hmac_sha256(key, data)

        assert result.shape == (32,)
        assert result.dtype == torch.uint8
```

**Step 2: Implement HMAC in Python (uses sha256 operator)**

```python
# src/torchscience/encryption/_hmac.py
import torch
from torch import Tensor

from torchscience.encryption._sha256 import sha256


def hmac_sha256(key: Tensor, data: Tensor) -> Tensor:
    """Compute HMAC-SHA256.

    Parameters
    ----------
    key : Tensor
        Key as (k,) uint8 tensor. Any length.
    data : Tensor
        Message as (..., n) uint8 tensor.

    Returns
    -------
    Tensor
        (..., 32) uint8 authentication tag.
    """
    block_size = 64

    # If key > block_size, hash it
    if key.size(0) > block_size:
        key = sha256(key)

    # Pad key to block_size
    if key.size(0) < block_size:
        key = torch.nn.functional.pad(key, (0, block_size - key.size(0)))

    # Inner and outer padding
    ipad = torch.full((block_size,), 0x36, dtype=torch.uint8, device=key.device)
    opad = torch.full((block_size,), 0x5C, dtype=torch.uint8, device=key.device)

    inner_key = key ^ ipad
    outer_key = key ^ opad

    # Inner hash: H(inner_key || data)
    inner_input = torch.cat([inner_key, data])
    inner_hash = sha256(inner_input)

    # Outer hash: H(outer_key || inner_hash)
    outer_input = torch.cat([outer_key, inner_hash])
    return sha256(outer_input)
```

**Step 3: Update __init__.py**

```python
from torchscience.encryption._hmac import hmac_sha256
# Add to __all__
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/torchscience/encryption/test__hmac.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/torchscience/encryption/_hmac.py src/torchscience/encryption/__init__.py tests/torchscience/encryption/test__hmac.py
git commit -m "feat(encryption): implement hmac_sha256"
```

---

## Phase 5: Privacy Module - Gaussian Mechanism

### Task 10: Create Privacy Module Structure

**Files:**
- Create: `src/torchscience/privacy/__init__.py`
- Create: `src/torchscience/privacy/_gaussian.py`
- Create: `tests/torchscience/privacy/__init__.py`
- Create: `tests/torchscience/privacy/test__gaussian.py`

**Step 1: Write failing tests**

```python
# tests/torchscience/privacy/__init__.py
```

```python
# tests/torchscience/privacy/test__gaussian.py
import pytest
import torch
import math

from torchscience.encryption import ChaCha20Generator
from torchscience.privacy import gaussian_mechanism


class TestGaussianMechanism:
    """Tests for Gaussian mechanism."""

    def test_output_shape(self):
        """Output has same shape as input."""
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, 20)

        result = gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_noise_scale(self):
        """Empirical noise scale matches theoretical."""
        gen = ChaCha20Generator(seed=42)
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5

        # Theoretical sigma
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        # Generate many samples
        x = torch.zeros(100000)
        noised = gaussian_mechanism(x, sensitivity, epsilon, delta, generator=gen)

        # Empirical std should be close to sigma
        empirical_std = noised.std().item()
        assert abs(empirical_std - sigma) < 0.1

    def test_mean_preserved(self):
        """Mean is approximately preserved (unbiased noise)."""
        gen = ChaCha20Generator(seed=42)
        x = torch.full((100000,), 5.0)

        noised = gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen)

        assert abs(noised.mean().item() - 5.0) < 0.1

    def test_determinism(self):
        """Same generator state produces same noise."""
        x = torch.randn(100)

        gen1 = ChaCha20Generator(seed=123)
        result1 = gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen1)

        gen2 = ChaCha20Generator(seed=123)
        result2 = gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen2)

        torch.testing.assert_close(result1, result2)

    def test_gradcheck(self):
        """Gradients flow through input."""
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)

        def func(x):
            return gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen)

        # Reset generator for reproducibility in gradcheck
        gen.manual_seed(42)
        torch.autograd.gradcheck(func, x)

    def test_gradgradcheck(self):
        """Second-order gradients work."""
        gen = ChaCha20Generator(seed=42)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)

        def func(x):
            return gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen)

        gen.manual_seed(42)
        torch.autograd.gradgradcheck(func, x)

    def test_higher_sensitivity_more_noise(self):
        """Higher sensitivity produces more noise."""
        gen1 = ChaCha20Generator(seed=42)
        gen2 = ChaCha20Generator(seed=42)
        x = torch.zeros(10000)

        noised_low = gaussian_mechanism(x, sensitivity=1.0, epsilon=1.0, delta=1e-5, generator=gen1)
        noised_high = gaussian_mechanism(x, sensitivity=10.0, epsilon=1.0, delta=1e-5, generator=gen2)

        assert noised_high.std() > noised_low.std()

    def test_higher_epsilon_less_noise(self):
        """Higher epsilon (less privacy) produces less noise."""
        gen1 = ChaCha20Generator(seed=42)
        gen2 = ChaCha20Generator(seed=42)
        x = torch.zeros(10000)

        noised_strict = gaussian_mechanism(x, sensitivity=1.0, epsilon=0.1, delta=1e-5, generator=gen1)
        noised_relaxed = gaussian_mechanism(x, sensitivity=1.0, epsilon=10.0, delta=1e-5, generator=gen2)

        assert noised_strict.std() > noised_relaxed.std()
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/torchscience/privacy/test__gaussian.py -v`
Expected: FAIL (privacy module not found)

**Step 3: Commit failing tests**

```bash
git add tests/torchscience/privacy/
git commit -m "test(privacy): add failing tests for gaussian_mechanism"
```

---

### Task 11: Implement Gaussian Mechanism

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`
- Create: `src/torchscience/csrc/kernel/privacy/gaussian_mechanism.h`
- Create: `src/torchscience/csrc/cpu/privacy/gaussian_mechanism.h`
- Create: `src/torchscience/csrc/meta/privacy/gaussian_mechanism.h`
- Create: `src/torchscience/csrc/autograd/privacy/gaussian_mechanism.h`
- Create: `src/torchscience/privacy/_gaussian.py`
- Create: `src/torchscience/privacy/__init__.py`

**Step 1: Add schema**

```cpp
// Privacy operators
m.def("gaussian_mechanism(Tensor x, Tensor noise, float sigma) -> Tensor");
m.def("gaussian_mechanism_backward(Tensor grad_output) -> Tensor");
```

**Step 2: Write kernel**

```cpp
// src/torchscience/csrc/kernel/privacy/gaussian_mechanism.h
#pragma once

namespace torchscience::kernel::privacy {

template <typename scalar_t>
inline scalar_t gaussian_mechanism_forward(scalar_t x, scalar_t noise, scalar_t sigma) {
    return x + sigma * noise;
}

// Backward: d(x + sigma * noise)/dx = 1
// So grad_input = grad_output

}  // namespace torchscience::kernel::privacy
```

**Step 3: Write CPU backend**

```cpp
// src/torchscience/csrc/cpu/privacy/gaussian_mechanism.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../../kernel/privacy/gaussian_mechanism.h"

namespace torchscience::cpu::privacy {

at::Tensor gaussian_mechanism(
    const at::Tensor& x,
    const at::Tensor& noise,
    double sigma
) {
    TORCH_CHECK(x.sizes() == noise.sizes(),
        "gaussian_mechanism: x and noise must have same shape");

    auto output = at::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gaussian_mechanism", [&] {
        auto x_data = x.data_ptr<scalar_t>();
        auto noise_data = noise.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        scalar_t sigma_t = static_cast<scalar_t>(sigma);

        for (int64_t i = 0; i < x.numel(); i++) {
            output_data[i] = kernel::privacy::gaussian_mechanism_forward(
                x_data[i], noise_data[i], sigma_t
            );
        }
    });

    return output;
}

at::Tensor gaussian_mechanism_backward(const at::Tensor& grad_output) {
    // d(x + noise)/dx = 1, so grad_input = grad_output
    return grad_output.clone();
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism);
    m.impl("gaussian_mechanism_backward", &gaussian_mechanism_backward);
}

}  // namespace torchscience::cpu::privacy
```

**Step 4: Write meta backend**

```cpp
// src/torchscience/csrc/meta/privacy/gaussian_mechanism.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::privacy {

at::Tensor gaussian_mechanism(
    const at::Tensor& x,
    const at::Tensor& noise,
    double sigma
) {
    return at::empty_like(x);
}

at::Tensor gaussian_mechanism_backward(const at::Tensor& grad_output) {
    return at::empty_like(grad_output);
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism);
    m.impl("gaussian_mechanism_backward", &gaussian_mechanism_backward);
}

}  // namespace torchscience::meta::privacy
```

**Step 5: Write autograd wrapper**

```cpp
// src/torchscience/csrc/autograd/privacy/gaussian_mechanism.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::autograd::privacy {

class GaussianMechanismFunction : public torch::autograd::Function<GaussianMechanismFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x,
        const at::Tensor& noise,
        double sigma
    ) {
        at::AutoDispatchBelowADInplaceOrView guard;
        return at::_ops::torchscience_gaussian_mechanism::call(x, noise, sigma);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto grad_output = grad_outputs[0];
        at::AutoDispatchBelowADInplaceOrView guard;
        auto grad_x = at::_ops::torchscience_gaussian_mechanism_backward::call(grad_output);
        return {grad_x, at::Tensor(), at::Tensor()};  // No grad for noise or sigma
    }
};

at::Tensor gaussian_mechanism_autograd(
    const at::Tensor& x,
    const at::Tensor& noise,
    double sigma
) {
    return GaussianMechanismFunction::apply(x, noise, sigma);
}

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism_autograd);
}

}  // namespace torchscience::autograd::privacy
```

**Step 6: Include in torchscience.cpp**

```cpp
#include "cpu/privacy/gaussian_mechanism.h"
#include "meta/privacy/gaussian_mechanism.h"
#include "autograd/privacy/gaussian_mechanism.h"
```

**Step 7: Write Python wrapper**

```python
# src/torchscience/privacy/_gaussian.py
import math

import torch
from torch import Tensor

from torchscience.encryption import ChaCha20Generator


def gaussian_mechanism(
    x: Tensor,
    sensitivity: float,
    epsilon: float,
    delta: float,
    generator: ChaCha20Generator,
) -> Tensor:
    """Add Gaussian noise calibrated for (epsilon, delta)-differential privacy.

    The noise scale is computed as:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Parameters
    ----------
    x : Tensor
        Input tensor to be privatized.
    sensitivity : float
        L2 sensitivity of the query.
    epsilon : float
        Privacy parameter epsilon (smaller = more private).
    delta : float
        Privacy parameter delta (probability of privacy breach).
    generator : ChaCha20Generator
        Cryptographically secure random number generator.

    Returns
    -------
    Tensor
        Input plus calibrated Gaussian noise.
    """
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = generator.randn(x.shape, x.dtype).to(x.device)
    return torch.ops.torchscience.gaussian_mechanism(x, noise, sigma)
```

**Step 8: Create __init__.py**

```python
# src/torchscience/privacy/__init__.py
from torchscience.privacy._gaussian import gaussian_mechanism

__all__ = [
    "gaussian_mechanism",
]
```

**Step 9: Run tests**

Run: `.venv/bin/python -m pytest tests/torchscience/privacy/test__gaussian.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/torchscience/privacy/ src/torchscience/csrc/
git commit -m "feat(privacy): implement gaussian_mechanism"
```

---

## Phase 6: Remaining Privacy Mechanisms

### Task 12: Implement Laplace Mechanism

Follow same pattern as Gaussian mechanism:

**Files:**
- Create: `src/torchscience/privacy/_laplace.py`
- Create: `tests/torchscience/privacy/test__laplace.py`
- Create: `src/torchscience/csrc/kernel/privacy/laplace_mechanism.h`
- Create: `src/torchscience/csrc/cpu/privacy/laplace_mechanism.h`
- Create: `src/torchscience/csrc/meta/privacy/laplace_mechanism.h`
- Create: `src/torchscience/csrc/autograd/privacy/laplace_mechanism.h`

**Key difference:** Noise scale is `b = sensitivity / epsilon` and uses Laplace distribution.

Laplace samples from uniform: `b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)` where u ~ Uniform(0,1).

---

### Task 13: Implement Exponential Mechanism

**Files:**
- Create: `src/torchscience/privacy/_exponential.py`
- Create: `tests/torchscience/privacy/test__exponential.py`

**Implementation:** Sample index proportional to `exp(epsilon * utility / (2 * sensitivity))` using Gumbel-max trick.

---

### Task 14: Implement Randomized Response

**Files:**
- Create: `src/torchscience/privacy/_randomized_response.py`
- Create: `tests/torchscience/privacy/test__randomized_response.py`

**Implementation:** Flip each value with probability `1 / (1 + exp(epsilon))`.

---

## Final Steps

### Task 15: Update Main Package Exports

**Files:**
- Modify: `src/torchscience/__init__.py`

Add imports for new modules so they're discoverable.

### Task 16: Final Integration Test

Run full test suite:

```bash
.venv/bin/python -m pytest tests/torchscience/encryption/ tests/torchscience/privacy/ -v
```

### Task 17: Commit and Merge

```bash
git add -A
git commit -m "feat: complete encryption and privacy modules"
```

Ready to merge back to main.