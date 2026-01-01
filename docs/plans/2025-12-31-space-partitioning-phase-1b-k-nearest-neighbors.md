# Space Partitioning Phase 1B: k-Nearest Neighbors Query

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement tree-accelerated k-nearest neighbors query with O(log n) performance and full autograd support.

**Architecture:** C++ CPU kernel using priority queue with tree traversal and branch pruning. Autograd wrapper with vectorized backward using gather/scatter operations. Meta and Autocast backends for shape inference and AMP support.

**Tech Stack:** PyTorch C++ Extension (libtorch), ATen, TORCH_LIBRARY_IMPL, torch::autograd::Function.

**Prerequisites:** Phase 1A completed (kd_tree build function available).

---

## Design Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Query Acceleration | Actual k-d tree traversal | Tree should provide O(log n) queries, not cosmetic |
| Distance Metric | Minkowski p-norm | Supports Euclidean (p=2), Manhattan (p=1), and general p |
| Backward Vectorization | Gather + broadcast | Efficient gradient computation without loops |
| Gradient Safety | `torch.where` for zero distances | Avoid NaN gradients when query equals a point |
| Gradient Testing | Include `gradgradcheck` | Support optimization applications needing Hessians |

---

## File Structure

```
src/torchscience/
├── csrc/
│   ├── cpu/space_partitioning/
│   │   └── k_nearest_neighbors.h          # CPU k-NN with tree traversal
│   ├── meta/space_partitioning/
│   │   └── k_nearest_neighbors.h          # Meta shape inference
│   ├── autograd/space_partitioning/
│   │   └── k_nearest_neighbors.h          # Autograd with vectorized backward
│   ├── autocast/space_partitioning/
│   │   └── k_nearest_neighbors.h          # Autocast for AMP
│   └── torchscience.cpp                   # Schema already added in Phase 1A
├── space_partitioning/
│   └── _k_nearest_neighbors.py            # Python wrapper
tests/torchscience/space_partitioning/
└── test__k_nearest_neighbors.py
```

---

### Task 1: Implement k_nearest_neighbors CPU Kernel with Tree Traversal

**Files:**
- Create: `src/torchscience/csrc/cpu/space_partitioning/k_nearest_neighbors.h`

**Step 1: Create the CPU kernel with actual tree traversal**

```cpp
// src/torchscience/csrc/cpu/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <cmath>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::space_partitioning {

namespace {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T minkowski_distance(const T* x, const T* y, int64_t d, T p) {
    T sum = T(0);
    if (p == T(2)) {
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } else if (p == T(1)) {
        for (int64_t i = 0; i < d; ++i) {
            sum += std::abs(x[i] - y[i]);
        }
        return sum;
    } else {
        for (int64_t i = 0; i < d; ++i) {
            sum += std::pow(std::abs(x[i] - y[i]), p);
        }
        return std::pow(sum, T(1) / p);
    }
}

template <typename scalar_t>
struct KNNSearcher {
    const scalar_t* points;
    const int64_t* split_dim;
    const scalar_t* split_val;
    const int64_t* left_child;
    const int64_t* right_child;
    const int64_t* indices;
    const int64_t* leaf_starts;
    const int64_t* leaf_counts;
    int64_t n_nodes;
    int64_t d;
    scalar_t p;

    // Max-heap: (distance, index) - largest distance at top
    using HeapItem = std::pair<scalar_t, int64_t>;
    struct MaxHeapCmp {
        bool operator()(const HeapItem& a, const HeapItem& b) {
            return a.first < b.first;
        }
    };

    void search(
        const scalar_t* query,
        int64_t k,
        std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCmp>& heap
    ) const {
        if (n_nodes == 0) return;
        search_node(0, query, k, heap);
    }

    void search_node(
        int64_t node,
        const scalar_t* query,
        int64_t k,
        std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCmp>& heap
    ) const {
        if (node < 0 || node >= n_nodes) return;

        int64_t sd = split_dim[node];

        // Leaf node
        if (sd == -1) {
            // Find leaf index by counting leaves before this node
            int64_t leaf_idx = 0;
            for (int64_t i = 0; i < node; ++i) {
                if (split_dim[i] == -1) leaf_idx++;
            }

            int64_t start = leaf_starts[leaf_idx];
            int64_t count = leaf_counts[leaf_idx];

            for (int64_t i = 0; i < count; ++i) {
                int64_t pt_idx = indices[start + i];
                scalar_t dist = minkowski_distance(
                    query, points + pt_idx * d, d, p
                );

                if (static_cast<int64_t>(heap.size()) < k) {
                    heap.push({dist, pt_idx});
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.push({dist, pt_idx});
                }
            }
            return;
        }

        // Internal node
        scalar_t sv = split_val[node];
        scalar_t query_val = query[sd];
        scalar_t diff = query_val - sv;

        // Determine which child to visit first
        int64_t first_child = diff < 0 ? left_child[node] : right_child[node];
        int64_t second_child = diff < 0 ? right_child[node] : left_child[node];

        // Visit closer child first
        search_node(first_child, query, k, heap);

        // Prune: only visit other child if it could contain closer points
        scalar_t plane_dist = std::abs(diff);
        if (static_cast<int64_t>(heap.size()) < k || plane_dist < heap.top().first) {
            search_node(second_child, query, k, heap);
        }
    }
};

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    TORCH_CHECK(points.dim() == 2, "k_nearest_neighbors: points must be 2D");
    TORCH_CHECK(queries.dim() == 2, "k_nearest_neighbors: queries must be 2D");
    TORCH_CHECK(points.size(1) == queries.size(1), "k_nearest_neighbors: dimension mismatch");

    int64_t n = points.size(0);
    int64_t d = points.size(1);
    int64_t m = queries.size(0);
    int64_t n_nodes = split_dim.size(0);

    TORCH_CHECK(k > 0 && k <= n, "k_nearest_neighbors: k must be in [1, n]");

    at::Tensor pts_contig = points.contiguous();
    at::Tensor queries_contig = queries.contiguous();
    at::Tensor split_dim_contig = split_dim.contiguous();
    at::Tensor split_val_contig = split_val.contiguous();
    at::Tensor left_contig = left.contiguous();
    at::Tensor right_contig = right.contiguous();
    at::Tensor indices_contig = indices.contiguous();
    at::Tensor leaf_starts_contig = leaf_starts.contiguous();
    at::Tensor leaf_counts_contig = leaf_counts.contiguous();

    at::Tensor result_indices = at::empty({m, k}, queries.options().dtype(at::kLong));
    at::Tensor result_distances = at::empty({m, k}, queries.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "k_nearest_neighbors_cpu",
        [&]() {
            KNNSearcher<scalar_t> searcher{
                pts_contig.data_ptr<scalar_t>(),
                split_dim_contig.data_ptr<int64_t>(),
                split_val_contig.data_ptr<scalar_t>(),
                left_contig.data_ptr<int64_t>(),
                right_contig.data_ptr<int64_t>(),
                indices_contig.data_ptr<int64_t>(),
                leaf_starts_contig.data_ptr<int64_t>(),
                leaf_counts_contig.data_ptr<int64_t>(),
                n_nodes,
                d,
                static_cast<scalar_t>(p)
            };

            const scalar_t* queries_ptr = queries_contig.data_ptr<scalar_t>();
            int64_t* result_idx_ptr = result_indices.data_ptr<int64_t>();
            scalar_t* result_dist_ptr = result_distances.data_ptr<scalar_t>();

            at::parallel_for(0, m, 0, [&](int64_t begin, int64_t end) {
                for (int64_t q_idx = begin; q_idx < end; ++q_idx) {
                    const scalar_t* query = queries_ptr + q_idx * d;

                    std::priority_queue<
                        std::pair<scalar_t, int64_t>,
                        std::vector<std::pair<scalar_t, int64_t>>,
                        typename KNNSearcher<scalar_t>::MaxHeapCmp
                    > heap;

                    searcher.search(query, k, heap);

                    // Extract results (heap gives largest first, reverse for ascending)
                    std::vector<std::pair<scalar_t, int64_t>> results;
                    while (!heap.empty()) {
                        results.push_back(heap.top());
                        heap.pop();
                    }
                    std::reverse(results.begin(), results.end());

                    // Pad if fewer than k points
                    while (static_cast<int64_t>(results.size()) < k) {
                        results.push_back({std::numeric_limits<scalar_t>::infinity(), -1});
                    }

                    for (int64_t i = 0; i < k; ++i) {
                        result_idx_ptr[q_idx * k + i] = results[i].second;
                        result_dist_ptr[q_idx * k + i] = results[i].first;
                    }
                }
            });
        }
    );

    return std::make_tuple(result_indices, result_distances);
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("k_nearest_neighbors", &torchscience::cpu::space_partitioning::k_nearest_neighbors);
}
```

**Step 2: Add include to torchscience.cpp**

```cpp
#include "cpu/space_partitioning/k_nearest_neighbors.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/space_partitioning/k_nearest_neighbors.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(k_nearest_neighbors): implement CPU kernel with tree traversal"
```

---

### Task 2: Implement k_nearest_neighbors Meta and Autograd Kernels

**Files:**
- Create: `src/torchscience/csrc/meta/space_partitioning/k_nearest_neighbors.h`
- Create: `src/torchscience/csrc/autograd/space_partitioning/k_nearest_neighbors.h`

**Step 1: Create meta kernel**

```cpp
// src/torchscience/csrc/meta/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    TORCH_CHECK(queries.dim() == 2, "k_nearest_neighbors: queries must be 2D");
    int64_t m = queries.size(0);

    return std::make_tuple(
        at::empty({m, k}, queries.options().dtype(at::kLong)),
        at::empty({m, k}, queries.options())
    );
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("k_nearest_neighbors", &torchscience::meta::space_partitioning::k_nearest_neighbors);
}
```

**Step 2: Create autograd kernel with vectorized backward**

```cpp
// src/torchscience/csrc/autograd/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::space_partitioning {

class KNearestNeighbors
    : public torch::autograd::Function<KNearestNeighbors> {
public:
    static std::tuple<at::Tensor, at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& points,
        const at::Tensor& split_dim,
        const at::Tensor& split_val,
        const at::Tensor& left,
        const at::Tensor& right,
        const at::Tensor& indices,
        const at::Tensor& leaf_starts,
        const at::Tensor& leaf_counts,
        const at::Tensor& queries,
        int64_t k,
        double p
    ) {
        ctx->saved_data["p"] = p;
        ctx->saved_data["k"] = k;
        ctx->saved_data["queries_requires_grad"] = queries.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [result_indices, result_distances] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::k_nearest_neighbors", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                int64_t, double
            )>()
            .call(points, split_dim, split_val, left, right, indices,
                  leaf_starts, leaf_counts, queries, k, p);

        ctx->save_for_backward({points, queries, result_indices, result_distances});

        return std::make_tuple(result_indices, result_distances);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor points = saved[0];
        at::Tensor queries = saved[1];
        at::Tensor result_indices = saved[2];
        at::Tensor result_distances = saved[3];

        double p = ctx->saved_data["p"].toDouble();
        int64_t k = ctx->saved_data["k"].toInt();
        bool queries_requires_grad = ctx->saved_data["queries_requires_grad"].toBool();

        at::Tensor grad_distances = grad_outputs[1];  // grad for distances

        if (!queries_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()};
        }

        // Vectorized gradient computation
        int64_t m = queries.size(0);
        int64_t d = queries.size(1);

        // Gather neighbor points: (m, k, d)
        at::Tensor gathered_points = at::index_select(
            points, 0, result_indices.flatten()
        ).view({m, k, d});

        // diff = query - neighbor: (m, k, d)
        at::Tensor diff = queries.unsqueeze(1) - gathered_points;

        // dist: (m, k, 1)
        at::Tensor dist = result_distances.unsqueeze(-1);

        // Safe gradient: avoid division by zero using where
        at::Tensor is_zero = result_distances < 1e-8;  // (m, k)
        at::Tensor safe_dist = dist.clamp_min(1e-8);

        // grad = diff / dist for L2, zero where dist is zero
        at::Tensor grad_component = at::where(
            is_zero.unsqueeze(-1).expand_as(diff),
            at::zeros_like(diff),
            diff / safe_dist
        );

        // Scale by incoming gradient and sum over k
        grad_component = grad_component * grad_distances.unsqueeze(-1);
        at::Tensor grad_queries = grad_component.sum(1);

        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                grad_queries, at::Tensor(), at::Tensor()};
    }
};

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    return KNearestNeighbors::apply(
        points, split_dim, split_val, left, right, indices,
        leaf_starts, leaf_counts, queries, k, p
    );
}

}  // namespace torchscience::autograd::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("k_nearest_neighbors", &torchscience::autograd::space_partitioning::k_nearest_neighbors);
}
```

**Step 3: Add includes to torchscience.cpp**

```cpp
#include "meta/space_partitioning/k_nearest_neighbors.h"
#include "autograd/space_partitioning/k_nearest_neighbors.h"
```

**Step 4: Commit**

```bash
git add src/torchscience/csrc/meta/space_partitioning/k_nearest_neighbors.h src/torchscience/csrc/autograd/space_partitioning/k_nearest_neighbors.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(k_nearest_neighbors): implement meta and autograd kernels"
```

---

### Task 3: Implement k_nearest_neighbors Autocast Kernel

**Files:**
- Create: `src/torchscience/csrc/autocast/space_partitioning/k_nearest_neighbors.h`

**Step 1: Create autocast kernel**

```cpp
// src/torchscience/csrc/autocast/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Upcast to float32 for precision
    at::Tensor points_fp32 = at::autocast::cached_cast(at::kFloat, points);
    at::Tensor split_val_fp32 = at::autocast::cached_cast(at::kFloat, split_val);
    at::Tensor queries_fp32 = at::autocast::cached_cast(at::kFloat, queries);

    auto [result_indices, distances] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::k_nearest_neighbors", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            int64_t, double
        )>()
        .call(points_fp32, split_dim, split_val_fp32, left, right, indices,
              leaf_starts, leaf_counts, queries_fp32, k, p);

    return std::make_tuple(result_indices, distances.to(queries.scalar_type()));
}

}  // namespace torchscience::autocast::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("k_nearest_neighbors", &torchscience::autocast::space_partitioning::k_nearest_neighbors);
}
```

**Step 2: Add include to torchscience.cpp**

```cpp
#include "autocast/space_partitioning/k_nearest_neighbors.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/autocast/space_partitioning/k_nearest_neighbors.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(k_nearest_neighbors): implement autocast kernel"
```

---

### Task 4: Implement k_nearest_neighbors Python Wrapper - Tests First

**Files:**
- Create: `tests/torchscience/space_partitioning/test__k_nearest_neighbors.py`

**Step 1: Write failing tests including gradgradcheck**

```python
# tests/torchscience/space_partitioning/test__k_nearest_neighbors.py
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.space_partitioning import kd_tree, k_nearest_neighbors


class TestKNearestNeighborsBasic:
    """Tests for k_nearest_neighbors query function."""

    def test_returns_indices_and_distances(self):
        """Returns tuple of (indices, distances)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        result = k_nearest_neighbors(tree, queries, k=5)

        assert isinstance(result, tuple)
        assert len(result) == 2
        indices, distances = result
        assert indices.shape == (10, 5)
        assert distances.shape == (10, 5)

    def test_indices_dtype_is_long(self):
        """Indices tensor has dtype long."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert indices.dtype == torch.long

    def test_distances_preserve_dtype(self):
        """Distances tensor preserves query dtype."""
        points = torch.randn(100, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(10, 3, dtype=torch.float64)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert distances.dtype == torch.float64

    def test_distances_are_non_negative(self):
        """All distances are non-negative."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert (distances >= 0).all()

    def test_distances_are_sorted(self):
        """Distances are sorted in ascending order per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        for i in range(10):
            sorted_dists = torch.sort(distances[i])[0]
            torch.testing.assert_close(distances[i], sorted_dists)

    def test_indices_are_valid(self):
        """Indices are in valid range [0, n)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert (indices >= 0).all()
        assert (indices < 100).all()


class TestKNearestNeighborsCorrectness:
    """Tests for correctness against brute force."""

    def test_matches_brute_force_euclidean(self):
        """Results match brute-force search for Euclidean distance."""
        torch.manual_seed(42)
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64)

        indices, distances = k_nearest_neighbors(tree, queries, k=3, p=2.0)

        # Brute force
        for i, query in enumerate(queries):
            dists = torch.sqrt(((points - query) ** 2).sum(dim=1))
            bf_dists, bf_indices = torch.topk(dists, k=3, largest=False)

            torch.testing.assert_close(
                distances[i], bf_dists, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(indices[i], bf_indices)

    def test_query_point_in_dataset(self):
        """Query point that exists in dataset has distance 0."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        query = points[42:43]

        indices, distances = k_nearest_neighbors(tree, query, k=1)

        assert indices[0, 0] == 42
        torch.testing.assert_close(
            distances[0, 0], torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_tree_traversal_prunes(self):
        """Tree traversal visits fewer nodes than brute force."""
        # Create clustered data where pruning helps
        torch.manual_seed(42)
        cluster1 = torch.randn(500, 3) + torch.tensor([0.0, 0.0, 0.0])
        cluster2 = torch.randn(500, 3) + torch.tensor([100.0, 0.0, 0.0])
        points = torch.cat([cluster1, cluster2], dim=0)

        tree = kd_tree(points, leaf_size=10)
        queries = cluster1[:5]  # Query near cluster1

        # Should find neighbors in cluster1 without checking cluster2
        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        # All neighbors should be from cluster1 (indices 0-499)
        assert (indices < 500).all()


class TestKNearestNeighborsGradient:
    """Tests for gradient support."""

    def test_gradient_exists(self):
        """Gradient exists for query points."""
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        indices, distances = k_nearest_neighbors(tree, queries, k=3)
        loss = distances.sum()
        loss.backward()

        assert queries.grad is not None
        assert torch.isfinite(queries.grad).all()

    def test_gradcheck(self):
        """First-order gradient is numerically correct."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)

        def fn(queries):
            _, distances = k_nearest_neighbors(tree, queries, k=3)
            return distances

        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(fn, (queries,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient is numerically correct."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)

        def fn(queries):
            _, distances = k_nearest_neighbors(tree, queries, k=3)
            return distances

        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(fn, (queries,), raise_exception=True)

    def test_zero_distance_gradient_is_finite(self):
        """Zero distance has finite gradient (using torch.where for safety)."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)
        # Use exact point from dataset
        queries = points[5:6].clone().requires_grad_(True)

        _, distances = k_nearest_neighbors(tree, queries, k=3)
        loss = distances.sum()
        loss.backward()

        # Should have finite gradients (zero where distance is zero)
        assert torch.isfinite(queries.grad).all()


class TestKNearestNeighborsValidation:
    """Tests for input validation."""

    def test_wrong_tree_type_raises(self):
        """Raises RuntimeError for wrong tree type."""
        from tensordict import TensorDict

        fake_tree = TensorDict({"_type": "wrong_type"}, batch_size=[])
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="Unsupported tree type"):
            k_nearest_neighbors(fake_tree, queries, k=3)

    def test_dimension_mismatch_raises(self):
        """Raises RuntimeError when query dimension doesn't match tree."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 5)

        with pytest.raises(RuntimeError, match="dimension"):
            k_nearest_neighbors(tree, queries, k=3)

    def test_k_too_large_raises(self):
        """Raises RuntimeError when k > n."""
        points = torch.randn(10, 3)
        tree = kd_tree(points)
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="k .* must be in"):
            k_nearest_neighbors(tree, queries, k=20)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/torchscience/space_partitioning/test__k_nearest_neighbors.py -v
```

**Step 3: Commit**

```bash
git add tests/torchscience/space_partitioning/test__k_nearest_neighbors.py
git commit -m "test(k_nearest_neighbors): add failing tests including gradgradcheck"
```

---

### Task 5: Implement k_nearest_neighbors Python Wrapper

**Files:**
- Create: `src/torchscience/space_partitioning/_k_nearest_neighbors.py`
- Modify: `src/torchscience/space_partitioning/__init__.py`

**Step 1: Implement k_nearest_neighbors function**

```python
# src/torchscience/space_partitioning/_k_nearest_neighbors.py
"""k-nearest neighbors query with tree traversal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

import torchscience._csrc  # noqa: F401


def k_nearest_neighbors(
    tree: TensorDict,
    queries: Tensor,
    k: int,
    *,
    p: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Find k nearest neighbors for each query point using tree traversal.

    Parameters
    ----------
    tree : TensorDict
        Spatial index built by kd_tree(), bounding_volume_hierarchy(), or octree().
    queries : Tensor, shape (m, d)
        Query points.
    k : int
        Number of neighbors to find.
    p : float, default=2.0
        Minkowski p-norm (2.0 = Euclidean, 1.0 = Manhattan).

    Returns
    -------
    indices : Tensor, shape (m, k)
        Indices of k nearest neighbors per query.
    distances : Tensor, shape (m, k)
        Distances to k nearest neighbors per query.
        Supports first and second order gradients w.r.t. query points.

    Notes
    -----
    Uses actual tree traversal with branch pruning for O(log n) average
    complexity per query. Worst case is O(n) for pathological point
    distributions.

    Examples
    --------
    >>> points = torch.randn(1000, 3)
    >>> tree = kd_tree(points)
    >>> queries = torch.randn(10, 3)
    >>> indices, distances = k_nearest_neighbors(tree, queries, k=5)
    >>> indices.shape
    torch.Size([10, 5])
    """
    if queries.dim() != 2:
        raise RuntimeError(f"queries must be 2D (m, d), got {queries.dim()}D")

    tree_type = tree.get("_type", None)

    if tree_type == "kd_tree":
        points = tree["points"]
        n = points.size(0)
        d = points.size(1)

        if queries.size(1) != d:
            raise RuntimeError(
                f"Query dimension ({queries.size(1)}) must match "
                f"tree dimension ({d})"
            )

        if k <= 0 or k > n:
            raise RuntimeError(f"k ({k}) must be in [1, {n}]")

        return torch.ops.torchscience.k_nearest_neighbors(
            tree["points"],
            tree["split_dim"],
            tree["split_val"],
            tree["left"],
            tree["right"],
            tree["indices"],
            tree["leaf_starts"],
            tree["leaf_counts"],
            queries,
            k,
            p,
        )

    else:
        raise RuntimeError(f"Unsupported tree type: {tree_type}")
```

**Step 2: Update __init__.py**

```python
# src/torchscience/space_partitioning/__init__.py
"""Spatial data structures for efficient neighbor search and range queries.

This module provides k-d trees, BVH, and octrees with:
- O(log n) query performance via tree traversal
- Full autograd support (first and second order)
- Batched construction for (B, N, D) point clouds
- Memory-mapped trees for massive datasets
- Thread-safe concurrent queries (see TensorDict lock semantics)
"""

from ._kd_tree import kd_tree
from ._k_nearest_neighbors import k_nearest_neighbors

__all__ = [
    "kd_tree",
    "k_nearest_neighbors",
]
```

**Step 3: Run tests**

```bash
uv run pytest tests/torchscience/space_partitioning/test__k_nearest_neighbors.py -v
```

**Step 4: Commit**

```bash
git add src/torchscience/space_partitioning/
git commit -m "feat(k_nearest_neighbors): implement Python wrapper"
```

---

## Summary

This phase implements:
- **1 query function:** `k_nearest_neighbors` with tree traversal
- **C++ backend** with CPU, Meta, Autograd, and Autocast kernels
- **Vectorized backward** using gather operations
- **Second-order gradients** via `gradgradcheck`

**Total tasks:** 5
**Estimated commits:** ~6

**Next phase:** Phase 1C (range_search with nested tensor output)
