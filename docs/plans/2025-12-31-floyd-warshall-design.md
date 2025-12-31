# Floyd-Warshall All-Pairs Shortest Paths Design

**Status:** Complete
**Date:** 2025-12-31

## Overview

Implement `torchscience.graph_theory.floyd_warshall` for computing all-pairs shortest paths using the Floyd-Warshall algorithm. The implementation includes both CPU and CUDA backends with full batching support.

## References

- [SciPy floyd_warshall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html)
- [NetworkX floyd_warshall](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.dense.floyd_warshall.html)
- [Boost floyd_warshall](https://www.boost.org/doc/libs/latest/libs/graph/doc/floyd_warshall_shortest.html)
- [Julia Graphs.jl](https://github.com/sbromberger/LightGraphs.jl/blob/master/src/shortestpaths/floyd-warshall.jl)
- [cp-algorithms](https://cp-algorithms.com/graph/all-pair-shortest-path-floyd-warshall.html)
- [CUDA blocked implementation](https://github.com/MTB90/cuda-floyd_warshall)

## API

```python
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
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Return format | `(distances, predecessors)` tuple | Matches SciPy, enables path reconstruction |
| Input format | Dense adjacency matrix + sparse COO | Dense is natural for O(N³) algorithm; sparse converted internally |
| Directed handling | `directed=True` default with parameter | Single function handles both cases |
| Negative cycles | Raise `NegativeCycleError` | Shortest paths undefined with negative cycles |
| Batching | Arbitrary leading dims `(*, N, N)` | Follows PyTorch conventions |
| Dtypes | float16, bfloat16, float32, float64 | Predecessors always int64 |
| CUDA | Blocked/tiled algorithm | 5-40x faster than naive |

## Algorithm

### CPU Implementation

```
1. If sparse input, convert to dense
2. If not directed, symmetrize: A = min(A, A.T)
3. Initialize:
   - dist = copy of adjacency matrix
   - pred[i,j] = i if edge exists, else -1
   - dist[i,i] = 0, pred[i,i] = -1
4. For k = 0 to N-1:
     For i = 0 to N-1:
       For j = 0 to N-1:
         if dist[i,k] + dist[k,j] < dist[i,j]:
           dist[i,j] = dist[i,k] + dist[k,j]
           pred[i,j] = pred[k,j]
5. Check diagonal: if any dist[i,i] < 0, raise NegativeCycleError
6. Return (dist, pred)
```

Batch parallelism via `at::parallel_for`.

### CUDA Blocked Algorithm

Divides N×N matrix into tiles of size B×B (B=32). Each iteration k processes tiles in three phases:

**Phase 1: Self-dependent tile**
- Process tile (k,k) that depends only on itself
- Single thread block, uses shared memory

**Phase 2: Row and column tiles**
- Process tiles in row k and column k (excluding diagonal)
- Each depends on phase 1 result + itself
- Multiple thread blocks in parallel

**Phase 3: All remaining tiles**
- Process all other tiles
- Each depends on its row-k tile and column-k tile from phase 2
- Fully parallel across all tiles

```
Iteration k:
┌───┬───┬───┬───┐
│   │ 2 │   │   │  Phase 1: tile (k,k)
├───┼───┼───┼───┤  Phase 2: row k, col k
│ 2 │ 1 │ 2 │ 2 │  Phase 3: everything else
├───┼───┼───┼───┤
│   │ 2 │   │   │
├───┼───┼───┼───┤
│   │ 2 │   │   │
└───┴───┴───┴───┘
     k
```

Batch dimension mapped to grid's z-dimension.

## File Structure

```
src/torchscience/
├── graph_theory/
│   ├── __init__.py
│   └── _floyd_warshall.py          # Python wrapper + NegativeCycleError
└── csrc/
    ├── cpu/graph_theory/
    │   └── floyd_warshall.h         # CPU kernel
    ├── cuda/graph_theory/
    │   └── floyd_warshall.cu        # CUDA blocked kernel
    ├── meta/graph_theory/
    │   └── floyd_warshall.h         # Shape inference
    └── torchscience.cpp             # Register operators

tests/torchscience/graph_theory/
    └── test__floyd_warshall.py
```

## Batching and Dtype Handling

**Batching:**
```python
batch_shape = input.shape[:-2]
N = input.shape[-1]
input_flat = input.reshape(-1, N, N)  # (B, N, N)
dist_flat, pred_flat = _impl(input_flat, directed)
distances = dist_flat.reshape(*batch_shape, N, N)
predecessors = pred_flat.reshape(*batch_shape, N, N)
```

**Dtype mapping:**

| Input dtype | Distance dtype | Predecessor dtype |
|-------------|----------------|-------------------|
| float16     | float16        | int64             |
| bfloat16    | bfloat16       | int64             |
| float32     | float32        | int64             |
| float64     | float64        | int64             |

**Sparse input:** Converted to dense via `input.to_dense()`.

## Error Handling

**Input Validation (Python):**
- Shape: must be at least 2D, last two dims must be equal
- Dtype: must be floating-point

**Negative Cycle Detection (C++):**
- After main loop, check if any `dist[i,i] < 0`
- Return error indicator to Python, which raises `NegativeCycleError`

**Edge Cases:**

| Case | Behavior |
|------|----------|
| Empty graph (N=0) | Return empty tensors `(*, 0, 0)` |
| Single node (N=1) | Return `[[0]]` distances, `[[-1]]` predecessors |
| No edges (all inf) | Distances stay inf, predecessors stay -1 |
| Self-loops with weight | Ignored (diagonal set to 0) |
| inf + inf | Stays inf (guarded arithmetic) |

**Inf Arithmetic Guard:**
```cpp
if (dist_ik < inf && dist_kj < inf) {
    scalar_t new_dist = dist_ik + dist_kj;
    if (new_dist < dist_ij) {
        dist_ij = new_dist;
        pred_ij = pred_kj;
    }
}
```

## Testing Strategy

```python
class TestFloydWarshallBasic:
    """Core functionality."""
    def test_simple_graph(self): ...
    def test_disconnected_nodes(self): ...
    def test_single_node(self): ...
    def test_empty_graph(self): ...
    def test_predecessor_reconstruction(self): ...

class TestFloydWarshallDirected:
    """Directed vs undirected."""
    def test_directed_asymmetric(self): ...
    def test_undirected_symmetrizes(self): ...

class TestFloydWarshallNegative:
    """Negative weights and cycles."""
    def test_negative_weights_no_cycle(self): ...
    def test_negative_cycle_raises(self): ...
    def test_negative_cycle_node_reported(self): ...

class TestFloydWarshallBatching:
    """Batch dimension handling."""
    def test_batch_2d(self): ...
    def test_batch_3d(self): ...
    def test_batch_4d(self): ...
    def test_batch_consistency(self): ...

class TestFloydWarshallDtypes:
    """Dtype support."""
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
                                        torch.float32, torch.float64])
    def test_dtype(self, dtype): ...

class TestFloydWarshallSparse:
    """Sparse COO input."""
    def test_sparse_matches_dense(self): ...

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFloydWarshallCUDA:
    """CUDA backend."""
    def test_cuda_matches_cpu(self): ...
    def test_cuda_batched(self): ...
    def test_cuda_large_graph(self): ...

class TestFloydWarshallReference:
    """Compare against SciPy."""
    def test_matches_scipy(self): ...
```

## Complexity

- **Time:** O(N³) per graph, O(B × N³) for batch of B graphs
- **Space:** O(N²) for distances + O(N²) for predecessors per graph
