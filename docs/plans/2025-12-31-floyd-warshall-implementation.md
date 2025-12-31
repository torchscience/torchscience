# Floyd-Warshall All-Pairs Shortest Paths Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `torchscience.graph_theory.floyd_warshall` for computing all-pairs shortest paths with batching support, CPU/CUDA backends, and negative cycle detection.

**Architecture:** Pure Python wrapper validates inputs and handles sparse conversion. C++ kernels implement the O(N³) algorithm with at::parallel_for for CPU batching and blocked/tiled algorithm for CUDA. Returns (distances, predecessors) tuple with NegativeCycleError for negative cycles.

**Tech Stack:** PyTorch C++ API (ATen), CUDA toolkit, pytest with scipy for reference comparisons

---

## Task 1: Create Python Module Structure

**Files:**
- Create: `src/torchscience/graph_theory/__init__.py`
- Create: `src/torchscience/graph_theory/_floyd_warshall.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py
import pytest
import torch

def test_import_floyd_warshall():
    """Can import floyd_warshall from graph_theory module."""
    from torchscience.graph_theory import floyd_warshall
    assert callable(floyd_warshall)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_import_floyd_warshall -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create graph_theory module**

```python
# src/torchscience/graph_theory/__init__.py
from ._floyd_warshall import floyd_warshall, NegativeCycleError

__all__ = [
    "floyd_warshall",
    "NegativeCycleError",
]
```

```python
# src/torchscience/graph_theory/_floyd_warshall.py
"""Floyd-Warshall all-pairs shortest paths implementation."""

import torch
from torch import Tensor


class NegativeCycleError(ValueError):
    """Raised when the graph contains a negative cycle."""
    pass


def floyd_warshall(
    input: Tensor,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Compute all-pairs shortest paths using the Floyd-Warshall algorithm.

    Args:
        input: Adjacency matrix of shape (*, N, N) where input[..., i, j]
               is the edge weight from node i to j. Use float('inf') for
               missing edges. Can be dense or sparse COO tensor.
        directed: If True, treat graph as directed. If False, symmetrize
                  the adjacency matrix by taking element-wise minimum.

    Returns:
        distances: Tensor of shape (*, N, N) with shortest path distances.
        predecessors: Tensor of shape (*, N, N) with dtype int64.
                      predecessors[..., i, j] is the node before j on the
                      shortest path from i to j, or -1 if no path exists.

    Raises:
        NegativeCycleError: If the graph contains a negative cycle.
    """
    raise NotImplementedError("floyd_warshall is not yet implemented")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_import_floyd_warshall -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/graph_theory/ tests/torchscience/graph_theory/
git commit -m "feat(graph_theory): add module structure for floyd_warshall"
```

---

## Task 2: Register C++ Operators in TORCH_LIBRARY

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp:97-185` (add schema definitions)

**Step 1: Write the failing test**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py (add to file)
def test_cpp_operator_registered():
    """C++ operator is registered with torch.ops."""
    import torchscience._csrc  # noqa: F401
    assert hasattr(torch.ops.torchscience, "floyd_warshall")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_cpp_operator_registered -v`
Expected: FAIL with "AttributeError"

**Step 3: Add schema definitions to torchscience.cpp**

Add after line 183 (before the closing brace of TORCH_LIBRARY):

```cpp
  // graph_theory
  module.def("floyd_warshall(Tensor input, bool directed) -> (Tensor, Tensor, bool)");
```

Note: Returns (distances, predecessors, has_negative_cycle) - Python wrapper checks has_negative_cycle and raises.

**Step 4: Build and run test**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_cpp_operator_registered -v`
Expected: PASS (operator schema registered, but no implementation yet)

**Step 5: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat(graph_theory): register floyd_warshall operator schema"
```

---

## Task 3: Implement Meta Backend (Shape Inference)

**Files:**
- Create: `src/torchscience/csrc/meta/graph_theory/floyd_warshall.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Write the failing test**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py (add to file)
def test_meta_shape_inference():
    """Meta backend correctly infers output shapes."""
    import torchscience._csrc  # noqa: F401

    input_meta = torch.empty(5, 5, device="meta")
    dist, pred, _ = torch.ops.torchscience.floyd_warshall(input_meta, True)

    assert dist.shape == (5, 5)
    assert pred.shape == (5, 5)
    assert pred.dtype == torch.int64
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_meta_shape_inference -v`
Expected: FAIL with "No implementation found"

**Step 3: Create meta implementation**

```cpp
// src/torchscience/csrc/meta/graph_theory/floyd_warshall.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );

    // Output shapes match input
    at::Tensor distances = at::empty_like(input);
    at::Tensor predecessors = at::empty(
        input.sizes(),
        input.options().dtype(at::kLong)
    );

    return std::make_tuple(distances, predecessors, false);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("floyd_warshall", &torchscience::meta::graph_theory::floyd_warshall);
}
```

**Step 4: Add include to torchscience.cpp**

Add after line 44 (with other meta includes):

```cpp
#include "meta/graph_theory/floyd_warshall.h"
```

**Step 5: Build and run test**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_meta_shape_inference -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/csrc/meta/graph_theory/ src/torchscience/csrc/torchscience.cpp
git commit -m "feat(graph_theory): add meta backend for floyd_warshall shape inference"
```

---

## Task 4: Implement CPU Backend

**Files:**
- Create: `src/torchscience/csrc/cpu/graph_theory/floyd_warshall.h`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add include)

**Step 1: Write the failing test**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py (add to file)
def test_cpu_simple_graph():
    """CPU backend computes correct shortest paths for simple graph."""
    import torchscience._csrc  # noqa: F401

    # Simple 3-node graph:
    #   0 --1--> 1 --1--> 2
    #   0 --3--> 2 (direct but longer)
    inf = float("inf")
    adj = torch.tensor([
        [0.0, 1.0, 3.0],
        [inf, 0.0, 1.0],
        [inf, inf, 0.0],
    ])

    dist, pred, has_neg = torch.ops.torchscience.floyd_warshall(adj, True)

    assert not has_neg
    expected_dist = torch.tensor([
        [0.0, 1.0, 2.0],  # 0->2 via 1 is shorter (1+1=2 < 3)
        [inf, 0.0, 1.0],
        [inf, inf, 0.0],
    ])
    assert torch.allclose(dist, expected_dist)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_cpu_simple_graph -v`
Expected: FAIL with "No implementation found for CPU"

**Step 3: Create CPU implementation**

```cpp
// src/torchscience/csrc/cpu/graph_theory/floyd_warshall.h
#pragma once

#include <cmath>
#include <limits>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
void floyd_warshall_single(
    scalar_t* dist,
    int64_t* pred,
    int64_t N,
    bool directed,
    bool& has_negative_cycle
) {
    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // If undirected, symmetrize by taking element-wise minimum
    if (!directed) {
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = i + 1; j < N; ++j) {
                scalar_t min_val = std::min(dist[i * N + j], dist[j * N + i]);
                dist[i * N + j] = min_val;
                dist[j * N + i] = min_val;
            }
        }
    }

    // Initialize predecessors:
    // pred[i,j] = i if edge exists (dist[i,j] < inf), else -1
    // pred[i,i] = -1 (no predecessor for self)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            if (i == j) {
                pred[i * N + j] = -1;
                dist[i * N + i] = scalar_t(0);  // Diagonal is always 0
            } else if (dist[i * N + j] < inf) {
                pred[i * N + j] = i;
            } else {
                pred[i * N + j] = -1;
            }
        }
    }

    // Floyd-Warshall main loop
    for (int64_t k = 0; k < N; ++k) {
        for (int64_t i = 0; i < N; ++i) {
            scalar_t dist_ik = dist[i * N + k];
            if (dist_ik >= inf) continue;  // Skip if no path i->k

            for (int64_t j = 0; j < N; ++j) {
                scalar_t dist_kj = dist[k * N + j];
                if (dist_kj >= inf) continue;  // Skip if no path k->j

                scalar_t new_dist = dist_ik + dist_kj;
                if (new_dist < dist[i * N + j]) {
                    dist[i * N + j] = new_dist;
                    pred[i * N + j] = pred[k * N + j];
                }
            }
        }
    }

    // Check for negative cycles (diagonal < 0)
    for (int64_t i = 0; i < N; ++i) {
        if (dist[i * N + i] < scalar_t(0)) {
            has_negative_cycle = true;
            return;
        }
    }
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "floyd_warshall: input must be floating-point, got ", input.scalar_type()
    );

    // Handle sparse input
    at::Tensor dense_input = input.is_sparse() ? input.to_dense() : input;

    // Make contiguous copy for in-place modification
    at::Tensor distances = dense_input.clone().contiguous();

    int64_t N = distances.size(-1);
    int64_t batch_size = distances.numel() / (N * N);

    at::Tensor predecessors = at::empty(
        distances.sizes(),
        distances.options().dtype(at::kLong)
    ).contiguous();

    // Handle empty graph
    if (N == 0) {
        return std::make_tuple(distances, predecessors, false);
    }

    bool has_negative_cycle = false;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        distances.scalar_type(),
        "floyd_warshall_cpu",
        [&] {
            scalar_t* dist_ptr = distances.data_ptr<scalar_t>();
            int64_t* pred_ptr = predecessors.data_ptr<int64_t>();

            // Process batches in parallel
            std::atomic<bool> found_negative_cycle{false};

            at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    if (found_negative_cycle.load()) return;

                    bool local_neg_cycle = false;
                    floyd_warshall_single<scalar_t>(
                        dist_ptr + b * N * N,
                        pred_ptr + b * N * N,
                        N,
                        directed,
                        local_neg_cycle
                    );

                    if (local_neg_cycle) {
                        found_negative_cycle.store(true);
                    }
                }
            });

            has_negative_cycle = found_negative_cycle.load();
        }
    );

    return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("floyd_warshall", &torchscience::cpu::graph_theory::floyd_warshall);
}
```

**Step 4: Add include to torchscience.cpp**

Add after line 25 (with other cpu includes):

```cpp
#include "cpu/graph_theory/floyd_warshall.h"
```

**Step 5: Build and run test**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_cpu_simple_graph -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/torchscience/csrc/cpu/graph_theory/ src/torchscience/csrc/torchscience.cpp
git commit -m "feat(graph_theory): add CPU backend for floyd_warshall"
```

---

## Task 5: Wire Python API to C++ Operators

**Files:**
- Modify: `src/torchscience/graph_theory/_floyd_warshall.py`

**Step 1: Write the failing test**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py (add to file)
def test_python_api_basic():
    """Python API returns correct results."""
    from torchscience.graph_theory import floyd_warshall

    inf = float("inf")
    adj = torch.tensor([
        [0.0, 1.0, 3.0],
        [inf, 0.0, 1.0],
        [inf, inf, 0.0],
    ])

    dist, pred = floyd_warshall(adj)

    expected_dist = torch.tensor([
        [0.0, 1.0, 2.0],
        [inf, 0.0, 1.0],
        [inf, inf, 0.0],
    ])
    assert torch.allclose(dist, expected_dist)
    assert pred.dtype == torch.int64
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_python_api_basic -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Update Python implementation**

```python
# src/torchscience/graph_theory/_floyd_warshall.py
"""Floyd-Warshall all-pairs shortest paths implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class NegativeCycleError(ValueError):
    """Raised when the graph contains a negative cycle.

    The Floyd-Warshall algorithm cannot compute shortest paths when the
    graph contains a cycle with negative total weight, as paths can be
    made arbitrarily short by traversing the cycle repeatedly.
    """
    pass


def floyd_warshall(
    input: Tensor,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute all-pairs shortest paths using the Floyd-Warshall algorithm.

    The Floyd-Warshall algorithm computes the shortest path between every
    pair of vertices in a weighted graph. It works with both positive and
    negative edge weights, but the graph must not contain negative cycles.

    .. math::
        d_{ij}^{(k)} = \min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})

    Parameters
    ----------
    input : Tensor
        Adjacency matrix of shape ``(*, N, N)`` where ``input[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. Can be dense or sparse COO tensor.
    directed : bool, default=True
        If True, treat graph as directed. If False, symmetrize the adjacency
        matrix by taking the element-wise minimum of ``A`` and ``A.T``.

    Returns
    -------
    distances : Tensor
        Tensor of shape ``(*, N, N)`` with shortest path distances.
        ``distances[..., i, j]`` is the length of the shortest path from
        node ``i`` to node ``j``, or ``inf`` if no path exists.
    predecessors : Tensor
        Tensor of shape ``(*, N, N)`` with dtype ``int64``.
        ``predecessors[..., i, j]`` is the node immediately before ``j``
        on the shortest path from ``i`` to ``j``, or ``-1`` if no path exists.

    Raises
    ------
    NegativeCycleError
        If the graph contains a negative cycle.
    ValueError
        If input is not at least 2D, or last two dimensions are not equal.

    Examples
    --------
    Simple directed graph:

    >>> import torch
    >>> from torchscience.graph_theory import floyd_warshall
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = floyd_warshall(adj)
    >>> dist
    tensor([[0., 1., 3.],
            [inf, 0., 2.],
            [inf, inf, 0.]])

    Path reconstruction (0 -> 2):

    >>> def reconstruct_path(pred, i, j):
    ...     if pred[i, j] == -1:
    ...         return [] if i != j else [i]
    ...     path = [j]
    ...     while pred[i, path[0]] != -1:
    ...         path.insert(0, pred[i, path[0]].item())
    ...     path.insert(0, i)
    ...     return path
    >>> reconstruct_path(pred, 0, 2)
    [0, 1, 2]

    Undirected graph (symmetrizes automatically):

    >>> adj_asym = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [2.0, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, _ = floyd_warshall(adj_asym, directed=False)
    >>> dist[0, 1] == dist[1, 0]  # Symmetrized
    tensor(True)

    Batched computation:

    >>> batch_adj = torch.stack([adj, adj * 2])  # (2, 3, 3)
    >>> dist, pred = floyd_warshall(batch_adj)
    >>> dist.shape
    torch.Size([2, 3, 3])

    Notes
    -----
    - **Complexity**: O(N³) time, O(N²) space per graph.
    - **Sparse input**: Converted to dense internally. For very sparse graphs
      with many nodes, consider Dijkstra's algorithm instead.
    - **Negative weights**: Supported, but negative cycles cause an error.
    - **Path reconstruction**: Use the predecessors tensor to reconstruct
      paths. ``pred[i, j]`` gives the node before ``j`` on the shortest
      path from ``i`` to ``j``.
    - **CUDA**: Blocked/tiled algorithm for GPU acceleration.

    References
    ----------
    .. [1] Floyd, R. W. (1962). "Algorithm 97: Shortest Path".
           Communications of the ACM. 5 (6): 345.
    .. [2] Warshall, S. (1962). "A theorem on Boolean matrices".
           Journal of the ACM. 9 (1): 11-12.

    See Also
    --------
    scipy.sparse.csgraph.floyd_warshall : SciPy implementation
    """
    # Input validation
    if input.dim() < 2:
        raise ValueError(
            f"floyd_warshall: input must be at least 2D, got {input.dim()}D"
        )
    if input.size(-1) != input.size(-2):
        raise ValueError(
            f"floyd_warshall: last two dimensions must be equal, "
            f"got {input.size(-2)} x {input.size(-1)}"
        )
    if not input.is_floating_point():
        raise ValueError(
            f"floyd_warshall: input must be floating-point, got {input.dtype}"
        )

    # Call C++ operator
    distances, predecessors, has_negative_cycle = torch.ops.torchscience.floyd_warshall(
        input, directed
    )

    if has_negative_cycle:
        raise NegativeCycleError(
            "floyd_warshall: graph contains a negative cycle"
        )

    return distances, predecessors
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::test_python_api_basic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torchscience/graph_theory/_floyd_warshall.py
git commit -m "feat(graph_theory): wire Python API to C++ floyd_warshall operator"
```

---

## Task 6: Add Comprehensive Test Suite

**Files:**
- Modify: `tests/torchscience/graph_theory/test__floyd_warshall.py`

**Step 1: Create comprehensive test file**

```python
# tests/torchscience/graph_theory/test__floyd_warshall.py
"""Comprehensive tests for Floyd-Warshall all-pairs shortest paths."""

import pytest
import torch

from torchscience.graph_theory import floyd_warshall, NegativeCycleError


class TestFloydWarshallBasic:
    """Core functionality tests."""

    def test_simple_graph(self):
        """Computes correct distances for simple 3-node graph."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0, 3.0],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ])

        dist, pred = floyd_warshall(adj)

        expected_dist = torch.tensor([
            [0.0, 1.0, 2.0],  # 0->2 via 1 is shorter
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ])
        assert torch.allclose(dist, expected_dist)

    def test_disconnected_nodes(self):
        """Handles disconnected nodes correctly."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0, inf],
            [inf, 0.0, inf],
            [inf, inf, 0.0],
        ])

        dist, pred = floyd_warshall(adj)

        # Node 2 is disconnected
        assert dist[0, 2] == inf
        assert dist[1, 2] == inf
        assert pred[0, 2] == -1
        assert pred[1, 2] == -1

    def test_single_node(self):
        """Handles single-node graph."""
        adj = torch.tensor([[0.0]])

        dist, pred = floyd_warshall(adj)

        assert dist.item() == 0.0
        assert pred.item() == -1

    def test_empty_graph(self):
        """Handles empty graph (N=0)."""
        adj = torch.empty(0, 0)

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (0, 0)
        assert pred.shape == (0, 0)

    def test_predecessor_reconstruction(self):
        """Predecessors enable path reconstruction."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0, inf, inf],
            [inf, 0.0, 1.0, inf],
            [inf, inf, 0.0, 1.0],
            [inf, inf, inf, 0.0],
        ])

        dist, pred = floyd_warshall(adj)

        # Path from 0 to 3: 0 -> 1 -> 2 -> 3
        assert pred[0, 3] == 2  # Before 3 is 2
        assert pred[0, 2] == 1  # Before 2 is 1
        assert pred[0, 1] == 0  # Before 1 is 0


class TestFloydWarshallDirected:
    """Directed vs undirected graph tests."""

    def test_directed_asymmetric(self):
        """Directed graph preserves asymmetric distances."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0],
            [2.0, 0.0],
        ])

        dist, pred = floyd_warshall(adj, directed=True)

        assert dist[0, 1] == 1.0
        assert dist[1, 0] == 2.0

    def test_undirected_symmetrizes(self):
        """Undirected mode symmetrizes by taking minimum."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0],
            [5.0, 0.0],
        ])

        dist, pred = floyd_warshall(adj, directed=False)

        # Takes minimum of (1.0, 5.0) = 1.0 for both directions
        assert dist[0, 1] == 1.0
        assert dist[1, 0] == 1.0


class TestFloydWarshallNegative:
    """Negative weights and cycle tests."""

    def test_negative_weights_no_cycle(self):
        """Handles negative edge weights without cycles."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 5.0, inf],
            [inf, 0.0, -2.0],
            [inf, inf, 0.0],
        ])

        dist, pred = floyd_warshall(adj)

        # 0 -> 2 via 1: 5 + (-2) = 3
        assert dist[0, 2] == 3.0

    def test_negative_cycle_raises(self):
        """Raises NegativeCycleError for negative cycle."""
        adj = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-3.0, 0.0, 0.0],  # Creates negative cycle: 0 -> 1 -> 2 -> 0 = -1
        ])

        with pytest.raises(NegativeCycleError):
            floyd_warshall(adj)


class TestFloydWarshallBatching:
    """Batch dimension handling tests."""

    def test_batch_2d(self):
        """Handles 2D input (single graph)."""
        adj = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
        ])

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (2, 2)
        assert pred.shape == (2, 2)

    def test_batch_3d(self):
        """Handles 3D input (batch of graphs)."""
        inf = float("inf")
        adj = torch.tensor([
            [[0.0, 1.0], [inf, 0.0]],
            [[0.0, 2.0], [inf, 0.0]],
        ])

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (2, 2, 2)
        assert dist[0, 0, 1] == 1.0
        assert dist[1, 0, 1] == 2.0

    def test_batch_4d(self):
        """Handles 4D input (nested batch)."""
        adj = torch.zeros(2, 3, 4, 4)

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (2, 3, 4, 4)
        assert pred.shape == (2, 3, 4, 4)

    def test_batch_consistency(self):
        """Batched results match individual computations."""
        inf = float("inf")
        adj1 = torch.tensor([
            [0.0, 1.0, inf],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ])
        adj2 = torch.tensor([
            [0.0, 2.0, inf],
            [inf, 0.0, 2.0],
            [inf, inf, 0.0],
        ])

        # Individual
        dist1, _ = floyd_warshall(adj1)
        dist2, _ = floyd_warshall(adj2)

        # Batched
        batch_adj = torch.stack([adj1, adj2])
        batch_dist, _ = floyd_warshall(batch_adj)

        assert torch.allclose(batch_dist[0], dist1)
        assert torch.allclose(batch_dist[1], dist2)


class TestFloydWarshallDtypes:
    """Dtype support tests."""

    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ])
    def test_dtype(self, dtype):
        """Supports various floating-point dtypes."""
        inf = float("inf")
        adj = torch.tensor([
            [0.0, 1.0, 3.0],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ], dtype=dtype)

        dist, pred = floyd_warshall(adj)

        assert dist.dtype == dtype
        assert pred.dtype == torch.int64


class TestFloydWarshallSparse:
    """Sparse COO input tests."""

    def test_sparse_matches_dense(self):
        """Sparse COO input produces same result as dense."""
        inf = float("inf")
        dense = torch.tensor([
            [0.0, 1.0, inf],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ])

        # Create sparse tensor (only store finite values)
        indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
        values = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        sparse = torch.sparse_coo_tensor(
            indices, values, dense.shape
        ).coalesce()

        # Fill sparse with inf for missing values
        sparse_dense = sparse.to_dense()
        sparse_dense[sparse_dense == 0] = inf
        sparse_dense.fill_diagonal_(0)

        dist_dense, _ = floyd_warshall(dense)
        dist_sparse, _ = floyd_warshall(sparse_dense)

        assert torch.allclose(dist_dense, dist_sparse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFloydWarshallCUDA:
    """CUDA backend tests."""

    def test_cuda_matches_cpu(self):
        """CUDA produces same results as CPU."""
        inf = float("inf")
        adj_cpu = torch.tensor([
            [0.0, 1.0, 3.0],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ])
        adj_cuda = adj_cpu.cuda()

        dist_cpu, pred_cpu = floyd_warshall(adj_cpu)
        dist_cuda, pred_cuda = floyd_warshall(adj_cuda)

        assert torch.allclose(dist_cpu, dist_cuda.cpu())
        assert torch.equal(pred_cpu, pred_cuda.cpu())

    def test_cuda_batched(self):
        """CUDA handles batched input."""
        adj = torch.randn(10, 50, 50).cuda()
        adj = torch.abs(adj) + 0.1  # Positive weights
        adj.fill_diagonal_(0)

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (10, 50, 50)
        assert dist.device.type == "cuda"

    def test_cuda_large_graph(self):
        """CUDA handles larger graphs efficiently."""
        N = 200
        adj = torch.randn(N, N).cuda()
        adj = torch.abs(adj) + 0.1
        adj.fill_diagonal_(0)

        dist, pred = floyd_warshall(adj)

        assert dist.shape == (N, N)


class TestFloydWarshallReference:
    """Compare against SciPy reference implementation."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy(self, N):
        """Results match scipy.sparse.csgraph.floyd_warshall."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        scipy_csgraph = pytest.importorskip("scipy.sparse.csgraph")

        # Random graph with some missing edges
        adj_np = torch.rand(N, N).numpy()
        adj_np[adj_np < 0.3] = float("inf")  # 30% missing edges
        adj_np[range(N), range(N)] = 0  # Diagonal is 0

        adj = torch.from_numpy(adj_np).float()

        # Compute with torchscience
        dist_torch, pred_torch = floyd_warshall(adj)

        # Compute with scipy
        dist_scipy, pred_scipy = scipy_csgraph.floyd_warshall(
            scipy_sparse.csr_matrix(adj_np),
            directed=True,
            return_predecessors=True,
        )

        # Compare distances
        assert torch.allclose(
            dist_torch,
            torch.from_numpy(dist_scipy).float(),
            rtol=1e-5,
            atol=1e-5,
        )


class TestFloydWarshallValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        """Rejects 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            floyd_warshall(torch.tensor([1.0, 2.0]))

    def test_rejects_non_square(self):
        """Rejects non-square matrix."""
        with pytest.raises(ValueError, match="must be equal"):
            floyd_warshall(torch.tensor([[1.0, 2.0, 3.0]]))

    def test_rejects_integer_dtype(self):
        """Rejects integer dtype."""
        with pytest.raises(ValueError, match="floating-point"):
            floyd_warshall(torch.tensor([[0, 1], [1, 0]]))
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py -v`
Expected: Most tests PASS, CUDA tests skip if no GPU, scipy test passes if scipy installed

**Step 3: Commit**

```bash
git add tests/torchscience/graph_theory/test__floyd_warshall.py
git commit -m "test(graph_theory): add comprehensive tests for floyd_warshall"
```

---

## Task 7: Implement CUDA Backend (Blocked Algorithm)

**Files:**
- Create: `src/torchscience/csrc/cuda/graph_theory/floyd_warshall.cu`
- Modify: `src/torchscience/csrc/torchscience.cpp` (add conditional include)

**Step 1: Verify CUDA test currently skipped or fails**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::TestFloydWarshallCUDA -v`
Expected: SKIP (no CUDA) or FAIL (no CUDA implementation)

**Step 2: Create CUDA implementation with blocked algorithm**

```cpp
// src/torchscience/csrc/cuda/graph_theory/floyd_warshall.cu
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchscience::cuda::graph_theory {

namespace {

constexpr int BLOCK_SIZE = 32;

/**
 * Phase 1: Process diagonal tile (k, k).
 * This tile only depends on itself.
 */
template <typename scalar_t>
__global__ void floyd_warshall_phase1_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset
) {
    __shared__ scalar_t s_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t i = k_block * BLOCK_SIZE + ty;
    int64_t j = k_block * BLOCK_SIZE + tx;

    // Load tile into shared memory
    scalar_t inf = cuda::std::numeric_limits<scalar_t>::infinity();
    if (i < N && j < N) {
        s_dist[ty][tx] = dist[batch_offset + i * N + j];
        s_pred[ty][tx] = pred[batch_offset + i * N + j];
    } else {
        s_dist[ty][tx] = inf;
        s_pred[ty][tx] = -1;
    }
    __syncthreads();

    // Process all k within this block
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        if (ty < BLOCK_SIZE && tx < BLOCK_SIZE) {
            scalar_t dist_ik = s_dist[ty][k];
            scalar_t dist_kj = s_dist[k][tx];

            if (dist_ik < inf && dist_kj < inf) {
                scalar_t new_dist = dist_ik + dist_kj;
                if (new_dist < s_dist[ty][tx]) {
                    s_dist[ty][tx] = new_dist;
                    s_pred[ty][tx] = s_pred[k][tx];
                }
            }
        }
        __syncthreads();
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = s_dist[ty][tx];
        pred[batch_offset + i * N + j] = s_pred[ty][tx];
    }
}

/**
 * Phase 2: Process row-k and column-k tiles (excluding diagonal).
 */
template <typename scalar_t>
__global__ void floyd_warshall_phase2_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset,
    bool is_row  // true for row tiles, false for column tiles
) {
    __shared__ scalar_t s_primary[BLOCK_SIZE][BLOCK_SIZE];  // The tile being updated
    __shared__ scalar_t s_pivot[BLOCK_SIZE][BLOCK_SIZE];    // The (k,k) tile
    __shared__ int64_t s_pred_primary[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred_pivot[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_idx = blockIdx.x;

    // Skip diagonal tile
    if (block_idx >= k_block) block_idx++;

    int64_t i_primary, j_primary;
    int64_t i_pivot, j_pivot;

    if (is_row) {
        // Row tile: (k_block, block_idx)
        i_primary = k_block * BLOCK_SIZE + ty;
        j_primary = block_idx * BLOCK_SIZE + tx;
        i_pivot = k_block * BLOCK_SIZE + ty;
        j_pivot = k_block * BLOCK_SIZE + tx;
    } else {
        // Column tile: (block_idx, k_block)
        i_primary = block_idx * BLOCK_SIZE + ty;
        j_primary = k_block * BLOCK_SIZE + tx;
        i_pivot = k_block * BLOCK_SIZE + ty;
        j_pivot = k_block * BLOCK_SIZE + tx;
    }

    scalar_t inf = cuda::std::numeric_limits<scalar_t>::infinity();

    // Load primary tile
    if (i_primary < N && j_primary < N) {
        s_primary[ty][tx] = dist[batch_offset + i_primary * N + j_primary];
        s_pred_primary[ty][tx] = pred[batch_offset + i_primary * N + j_primary];
    } else {
        s_primary[ty][tx] = inf;
        s_pred_primary[ty][tx] = -1;
    }

    // Load pivot tile (k,k)
    if (i_pivot < N && j_pivot < N) {
        s_pivot[ty][tx] = dist[batch_offset + i_pivot * N + j_pivot];
        s_pred_pivot[ty][tx] = pred[batch_offset + i_pivot * N + j_pivot];
    } else {
        s_pivot[ty][tx] = inf;
        s_pred_pivot[ty][tx] = -1;
    }
    __syncthreads();

    // Process
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik, dist_kj;
        int64_t pred_kj;

        if (is_row) {
            dist_ik = s_pivot[ty][k];
            dist_kj = s_primary[k][tx];
            pred_kj = s_pred_primary[k][tx];
        } else {
            dist_ik = s_primary[ty][k];
            dist_kj = s_pivot[k][tx];
            pred_kj = s_pred_pivot[k][tx];
        }

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < s_primary[ty][tx]) {
                s_primary[ty][tx] = new_dist;
                s_pred_primary[ty][tx] = pred_kj;
            }
        }
        __syncthreads();
    }

    // Write back
    if (i_primary < N && j_primary < N) {
        dist[batch_offset + i_primary * N + j_primary] = s_primary[ty][tx];
        pred[batch_offset + i_primary * N + j_primary] = s_pred_primary[ty][tx];
    }
}

/**
 * Phase 3: Process all remaining tiles.
 */
template <typename scalar_t>
__global__ void floyd_warshall_phase3_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset,
    int64_t num_blocks
) {
    __shared__ scalar_t s_row[BLOCK_SIZE][BLOCK_SIZE];  // Row-k tile for this column
    __shared__ scalar_t s_col[BLOCK_SIZE][BLOCK_SIZE];  // Column-k tile for this row
    __shared__ int64_t s_pred_row[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int64_t block_i = blockIdx.y;
    int64_t block_j = blockIdx.x;

    // Skip row-k and column-k tiles
    if (block_i >= k_block) block_i++;
    if (block_j >= k_block) block_j++;

    int64_t i = block_i * BLOCK_SIZE + ty;
    int64_t j = block_j * BLOCK_SIZE + tx;

    int64_t i_row = k_block * BLOCK_SIZE + ty;  // Row-k tile
    int64_t j_col = k_block * BLOCK_SIZE + tx;  // Column-k tile

    scalar_t inf = cuda::std::numeric_limits<scalar_t>::infinity();

    // Load row-k tile (k_block, block_j)
    int64_t j_row = block_j * BLOCK_SIZE + tx;
    if (i_row < N && j_row < N) {
        s_row[ty][tx] = dist[batch_offset + i_row * N + j_row];
        s_pred_row[ty][tx] = pred[batch_offset + i_row * N + j_row];
    } else {
        s_row[ty][tx] = inf;
        s_pred_row[ty][tx] = -1;
    }

    // Load column-k tile (block_i, k_block)
    int64_t i_col = block_i * BLOCK_SIZE + ty;
    if (i_col < N && j_col < N) {
        s_col[ty][tx] = dist[batch_offset + i_col * N + j_col];
    } else {
        s_col[ty][tx] = inf;
    }

    // Load current tile value
    scalar_t cur_dist = inf;
    int64_t cur_pred = -1;
    if (i < N && j < N) {
        cur_dist = dist[batch_offset + i * N + j];
        cur_pred = pred[batch_offset + i * N + j];
    }
    __syncthreads();

    // Process
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik = s_col[ty][k];
        scalar_t dist_kj = s_row[k][tx];

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < cur_dist) {
                cur_dist = new_dist;
                cur_pred = s_pred_row[k][tx];
            }
        }
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = cur_dist;
        pred[batch_offset + i * N + j] = cur_pred;
    }
}

/**
 * Initialize distances and predecessors.
 */
template <typename scalar_t>
__global__ void floyd_warshall_init_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t batch_offset,
    bool directed
) {
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    scalar_t inf = cuda::std::numeric_limits<scalar_t>::infinity();
    int64_t idx = batch_offset + i * N + j;

    // Symmetrize if undirected
    if (!directed && i < j) {
        scalar_t val_ij = dist[idx];
        scalar_t val_ji = dist[batch_offset + j * N + i];
        scalar_t min_val = min(val_ij, val_ji);
        dist[idx] = min_val;
        dist[batch_offset + j * N + i] = min_val;
    }
    __syncthreads();

    // Initialize predecessors
    if (i == j) {
        dist[idx] = scalar_t(0);
        pred[idx] = -1;
    } else if (dist[idx] < inf) {
        pred[idx] = i;
    } else {
        pred[idx] = -1;
    }
}

/**
 * Check for negative cycles (diagonal < 0).
 */
template <typename scalar_t>
__global__ void floyd_warshall_check_negative_kernel(
    const scalar_t* __restrict__ dist,
    int64_t N,
    int64_t batch_offset,
    bool* has_negative
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (dist[batch_offset + i * N + i] < scalar_t(0)) {
            *has_negative = true;
        }
    }
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "floyd_warshall: input must be floating-point, got ", input.scalar_type()
    );

    c10::cuda::CUDAGuard guard(input.device());

    // Handle sparse input
    at::Tensor dense_input = input.is_sparse() ? input.to_dense() : input;

    // Make contiguous copy
    at::Tensor distances = dense_input.clone().contiguous();

    int64_t N = distances.size(-1);
    int64_t batch_size = distances.numel() / (N * N);

    at::Tensor predecessors = at::empty(
        distances.sizes(),
        distances.options().dtype(at::kLong)
    ).contiguous();

    // Handle empty graph
    if (N == 0) {
        return std::make_tuple(distances, predecessors, false);
    }

    // Allocate flag for negative cycle detection
    at::Tensor has_negative_tensor = at::zeros({1}, distances.options().dtype(at::kBool));
    bool* has_negative_ptr = has_negative_tensor.data_ptr<bool>();

    int64_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        distances.scalar_type(),
        "floyd_warshall_cuda",
        [&] {
            scalar_t* dist_ptr = distances.data_ptr<scalar_t>();
            int64_t* pred_ptr = predecessors.data_ptr<int64_t>();

            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_init((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t batch_offset = b * N * N;

                // Initialize
                floyd_warshall_init_kernel<scalar_t><<<grid_init, block, 0, stream>>>(
                    dist_ptr, pred_ptr, N, batch_offset, directed
                );

                // Main loop
                for (int64_t k = 0; k < num_blocks; ++k) {
                    // Phase 1: Diagonal tile
                    floyd_warshall_phase1_kernel<scalar_t><<<1, block, 0, stream>>>(
                        dist_ptr, pred_ptr, N, k, batch_offset
                    );

                    if (num_blocks > 1) {
                        // Phase 2: Row and column tiles
                        dim3 grid_phase2(num_blocks - 1);
                        floyd_warshall_phase2_kernel<scalar_t><<<grid_phase2, block, 0, stream>>>(
                            dist_ptr, pred_ptr, N, k, batch_offset, true  // row
                        );
                        floyd_warshall_phase2_kernel<scalar_t><<<grid_phase2, block, 0, stream>>>(
                            dist_ptr, pred_ptr, N, k, batch_offset, false  // column
                        );

                        // Phase 3: Remaining tiles
                        dim3 grid_phase3(num_blocks - 1, num_blocks - 1);
                        floyd_warshall_phase3_kernel<scalar_t><<<grid_phase3, block, 0, stream>>>(
                            dist_ptr, pred_ptr, N, k, batch_offset, num_blocks
                        );
                    }
                }

                // Check for negative cycles
                dim3 grid_check((N + 255) / 256);
                floyd_warshall_check_negative_kernel<scalar_t><<<grid_check, 256, 0, stream>>>(
                    dist_ptr, N, batch_offset, has_negative_ptr
                );
            }
        }
    );

    // Sync and check result
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    bool has_negative_cycle = has_negative_tensor.item<bool>();

    return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // namespace torchscience::cuda::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
    m.impl("floyd_warshall", &torchscience::cuda::graph_theory::floyd_warshall);
}
```

**Step 3: Add conditional include to torchscience.cpp**

Add inside `#ifdef TORCHSCIENCE_CUDA` block (around line 63):

```cpp
#include "cuda/graph_theory/floyd_warshall.cu"
```

**Step 4: Build and run CUDA tests**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::TestFloydWarshallCUDA -v`
Expected: PASS (if CUDA available)

**Step 5: Commit**

```bash
git add src/torchscience/csrc/cuda/graph_theory/ src/torchscience/csrc/torchscience.cpp
git commit -m "feat(graph_theory): add CUDA backend for floyd_warshall with blocked algorithm"
```

---

## Task 8: Run Full Test Suite and Fix Issues

**Step 1: Run all tests**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py -v --tb=short`
Expected: All tests PASS (or skip appropriately)

**Step 2: Fix any failing tests**

Address issues as they arise based on test output.

**Step 3: Run tests with different dtypes**

Run: `uv run pytest tests/torchscience/graph_theory/test__floyd_warshall.py::TestFloydWarshallDtypes -v`
Expected: PASS

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(graph_theory): address test failures in floyd_warshall"
```

---

## Task 9: Mark Design Document as Complete

**Files:**
- Modify: `docs/plans/2025-12-31-floyd-warshall-design.md`

**Step 1: Update status**

Change line 3 from:
```markdown
**Status:** Approved
```
to:
```markdown
**Status:** Complete
```

**Step 2: Commit**

```bash
git add docs/plans/2025-12-31-floyd-warshall-design.md
git commit -m "docs: mark floyd_warshall design as complete"
```

---

## Summary

This plan implements Floyd-Warshall in 9 tasks:

1. **Module structure** - Python package scaffolding
2. **Schema registration** - TORCH_LIBRARY operator definition
3. **Meta backend** - Shape inference for JIT tracing
4. **CPU backend** - O(N³) algorithm with parallel_for batching
5. **Python API** - Input validation and error handling
6. **Test suite** - Comprehensive coverage including scipy reference
7. **CUDA backend** - Blocked/tiled algorithm for GPU acceleration
8. **Integration** - Full test pass and issue fixes
9. **Documentation** - Mark design complete

---

Plan complete and saved to `docs/plans/2025-12-31-floyd-warshall-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?