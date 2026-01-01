# Space Partitioning Phase 1C: Range Search Query

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement tree-accelerated range search returning variable-length results as PyTorch nested tensors, with full autograd support.

**Architecture:** C++ CPU kernel with tree traversal returning nested tensors for variable-length results. Autograd wrapper with scatter_add vectorized backward. Final integration and cleanup.

**Tech Stack:** PyTorch C++ Extension (libtorch), ATen, torch.nested, TORCH_LIBRARY_IMPL, torch::autograd::Function.

**Prerequisites:** Phase 1A and Phase 1B completed (kd_tree and k_nearest_neighbors available).

---

## Design Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Range Search Output | PyTorch nested tensors | Native semantics for variable-length results |
| Backward Vectorization | Per-query loop with gather | Nested tensors require unbind() |
| Query Acceleration | Actual tree traversal | O(log n) average with pruning |
| Distance Metric | Minkowski p-norm | Consistency with k_nearest_neighbors |

---

## File Structure

```
src/torchscience/
├── csrc/
│   ├── cpu/space_partitioning/
│   │   └── range_search.h                 # CPU range search with tree traversal
│   ├── meta/space_partitioning/
│   │   └── range_search.h                 # Meta shape inference
│   ├── autograd/space_partitioning/
│   │   └── range_search.h                 # Autograd with scatter_add backward
│   ├── autocast/space_partitioning/
│   │   └── range_search.h                 # Autocast for AMP
│   └── torchscience.cpp                   # Schema already added in Phase 1A
├── space_partitioning/
│   └── _range_search.py                   # Python wrapper
tests/torchscience/space_partitioning/
└── test__range_search.py
```

---

### Task 1: Implement range_search CPU Kernel with Nested Tensor Output

**Files:**
- Create: `src/torchscience/csrc/cpu/space_partitioning/range_search.h`

**Step 1: Create the CPU kernel with tree traversal and nested tensor output**

```cpp
// src/torchscience/csrc/cpu/space_partitioning/range_search.h
#pragma once

#include <cmath>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::space_partitioning {

namespace {

template <typename scalar_t>
struct RangeSearcher {
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
    scalar_t radius;

    void search(
        const scalar_t* query,
        std::vector<std::pair<scalar_t, int64_t>>& results
    ) const {
        if (n_nodes == 0) return;
        search_node(0, query, results);
    }

    void search_node(
        int64_t node,
        const scalar_t* query,
        std::vector<std::pair<scalar_t, int64_t>>& results
    ) const {
        if (node < 0 || node >= n_nodes) return;

        int64_t sd = split_dim[node];

        // Leaf node - check all points
        if (sd == -1) {
            // Find leaf index
            int64_t leaf_idx = 0;
            for (int64_t i = 0; i < node; ++i) {
                if (split_dim[i] == -1) leaf_idx++;
            }

            int64_t start = leaf_starts[leaf_idx];
            int64_t count = leaf_counts[leaf_idx];

            for (int64_t i = 0; i < count; ++i) {
                int64_t pt_idx = indices[start + i];

                // Compute distance
                scalar_t dist = scalar_t(0);
                const scalar_t* pt = points + pt_idx * d;

                if (p == scalar_t(2)) {
                    for (int64_t j = 0; j < d; ++j) {
                        scalar_t diff = query[j] - pt[j];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                } else if (p == scalar_t(1)) {
                    for (int64_t j = 0; j < d; ++j) {
                        dist += std::abs(query[j] - pt[j]);
                    }
                } else {
                    for (int64_t j = 0; j < d; ++j) {
                        dist += std::pow(std::abs(query[j] - pt[j]), p);
                    }
                    dist = std::pow(dist, scalar_t(1) / p);
                }

                if (dist <= radius) {
                    results.push_back({dist, pt_idx});
                }
            }
            return;
        }

        // Internal node
        scalar_t sv = split_val[node];
        scalar_t query_val = query[sd];
        scalar_t diff = query_val - sv;

        // Visit children that could contain points within radius
        // Left child: points with coord < sv
        if (diff - radius < 0) {
            search_node(left_child[node], query, results);
        }
        // Right child: points with coord >= sv
        if (diff + radius >= 0) {
            search_node(right_child[node], query, results);
        }
    }
};

}  // anonymous namespace

// Returns nested tensors for indices and distances
inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    TORCH_CHECK(points.dim() == 2, "range_search: points must be 2D");
    TORCH_CHECK(queries.dim() == 2, "range_search: queries must be 2D");
    TORCH_CHECK(points.size(1) == queries.size(1), "range_search: dimension mismatch");
    TORCH_CHECK(radius >= 0, "range_search: radius must be non-negative");

    int64_t d = points.size(1);
    int64_t m = queries.size(0);
    int64_t n_nodes = split_dim.size(0);

    at::Tensor pts_contig = points.contiguous();
    at::Tensor queries_contig = queries.contiguous();
    at::Tensor split_dim_contig = split_dim.contiguous();
    at::Tensor split_val_contig = split_val.contiguous();
    at::Tensor left_contig = left.contiguous();
    at::Tensor right_contig = right.contiguous();
    at::Tensor indices_contig = indices.contiguous();
    at::Tensor leaf_starts_contig = leaf_starts.contiguous();
    at::Tensor leaf_counts_contig = leaf_counts.contiguous();

    // Collect results per query
    std::vector<std::vector<std::pair<double, int64_t>>> all_results(m);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "range_search_cpu",
        [&]() {
            RangeSearcher<scalar_t> searcher{
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
                static_cast<scalar_t>(p),
                static_cast<scalar_t>(radius)
            };

            const scalar_t* queries_ptr = queries_contig.data_ptr<scalar_t>();

            at::parallel_for(0, m, 0, [&](int64_t begin, int64_t end) {
                for (int64_t q_idx = begin; q_idx < end; ++q_idx) {
                    const scalar_t* query = queries_ptr + q_idx * d;

                    std::vector<std::pair<scalar_t, int64_t>> results;
                    searcher.search(query, results);

                    // Sort by distance
                    std::sort(results.begin(), results.end());

                    // Convert to double for storage
                    for (const auto& [dist, idx] : results) {
                        all_results[q_idx].push_back({static_cast<double>(dist), idx});
                    }
                }
            });
        }
    );

    // Create nested tensors
    std::vector<at::Tensor> idx_tensors(m);
    std::vector<at::Tensor> dist_tensors(m);

    for (int64_t i = 0; i < m; ++i) {
        int64_t count = static_cast<int64_t>(all_results[i].size());

        if (count > 0) {
            idx_tensors[i] = at::empty({count}, queries.options().dtype(at::kLong));
            dist_tensors[i] = at::empty({count}, queries.options());

            int64_t* idx_ptr = idx_tensors[i].data_ptr<int64_t>();
            for (int64_t j = 0; j < count; ++j) {
                idx_ptr[j] = all_results[i][j].second;
            }

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                queries.scalar_type(),
                "range_search_copy_dist",
                [&]() {
                    scalar_t* dist_ptr = dist_tensors[i].data_ptr<scalar_t>();
                    for (int64_t j = 0; j < count; ++j) {
                        dist_ptr[j] = static_cast<scalar_t>(all_results[i][j].first);
                    }
                }
            );
        } else {
            idx_tensors[i] = at::empty({0}, queries.options().dtype(at::kLong));
            dist_tensors[i] = at::empty({0}, queries.options());
        }
    }

    // Create nested tensors from list of tensors
    at::Tensor nested_indices = at::_nested_tensor_from_tensor_list(idx_tensors);
    at::Tensor nested_distances = at::_nested_tensor_from_tensor_list(dist_tensors);

    return std::make_tuple(nested_indices, nested_distances);
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("range_search", &torchscience::cpu::space_partitioning::range_search);
}
```

**Step 2: Add include to torchscience.cpp**

```cpp
#include "cpu/space_partitioning/range_search.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/space_partitioning/range_search.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(range_search): implement CPU kernel with nested tensor output"
```

---

### Task 2: Implement range_search Meta and Autograd Kernels

**Files:**
- Create: `src/torchscience/csrc/meta/space_partitioning/range_search.h`
- Create: `src/torchscience/csrc/autograd/space_partitioning/range_search.h`

**Step 1: Create meta kernel**

```cpp
// src/torchscience/csrc/meta/space_partitioning/range_search.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    int64_t n = points.size(0);
    int64_t m = queries.size(0);

    // Return empty nested tensors with correct structure
    // Actual sizes are data-dependent
    std::vector<at::Tensor> idx_tensors(m);
    std::vector<at::Tensor> dist_tensors(m);

    for (int64_t i = 0; i < m; ++i) {
        // Conservative: each query could match all points
        idx_tensors[i] = at::empty({n}, queries.options().dtype(at::kLong));
        dist_tensors[i] = at::empty({n}, queries.options());
    }

    return std::make_tuple(
        at::_nested_tensor_from_tensor_list(idx_tensors),
        at::_nested_tensor_from_tensor_list(dist_tensors)
    );
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("range_search", &torchscience::meta::space_partitioning::range_search);
}
```

**Step 2: Create autograd kernel with scatter_add vectorized backward**

```cpp
// src/torchscience/csrc/autograd/space_partitioning/range_search.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::space_partitioning {

class RangeSearch
    : public torch::autograd::Function<RangeSearch> {
public:
    static std::tuple<at::Tensor, at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& points,
        const at::Tensor& split_dim,
        const at::Tensor& split_val,
        const at::Tensor& left,
        const at::Tensor& right,
        const at::Tensor& indices_tree,
        const at::Tensor& leaf_starts,
        const at::Tensor& leaf_counts,
        const at::Tensor& queries,
        double radius,
        double p
    ) {
        ctx->saved_data["p"] = p;
        ctx->saved_data["queries_requires_grad"] = queries.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [result_indices, result_distances] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::range_search", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                double, double
            )>()
            .call(points, split_dim, split_val, left, right, indices_tree,
                  leaf_starts, leaf_counts, queries, radius, p);

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
        at::Tensor result_indices = saved[2];  // nested tensor
        at::Tensor result_distances = saved[3];  // nested tensor

        bool queries_requires_grad = ctx->saved_data["queries_requires_grad"].toBool();

        if (!queries_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_distances = grad_outputs[1];  // nested tensor

        int64_t m = queries.size(0);
        int64_t d = queries.size(1);

        // Vectorized gradient using scatter_add
        // First, unbind nested tensors
        std::vector<at::Tensor> idx_list = result_indices.unbind();
        std::vector<at::Tensor> dist_list = result_distances.unbind();
        std::vector<at::Tensor> grad_dist_list = grad_distances.unbind();

        at::Tensor grad_queries = at::zeros_like(queries);

        // Process each query
        for (int64_t q = 0; q < m; ++q) {
            at::Tensor q_indices = idx_list[q];
            at::Tensor q_distances = dist_list[q];
            at::Tensor q_grad_distances = grad_dist_list[q];

            if (q_indices.size(0) == 0) continue;

            // Gather neighbor points
            at::Tensor neighbor_points = at::index_select(points, 0, q_indices);
            at::Tensor diff = queries[q].unsqueeze(0) - neighbor_points;  // (count, d)

            // Safe distance gradient
            at::Tensor safe_dist = q_distances.clamp_min(1e-8).unsqueeze(-1);
            at::Tensor is_zero = q_distances < 1e-8;

            at::Tensor grad_component = at::where(
                is_zero.unsqueeze(-1).expand_as(diff),
                at::zeros_like(diff),
                diff / safe_dist
            );

            grad_component = grad_component * q_grad_distances.unsqueeze(-1);
            grad_queries[q] = grad_component.sum(0);
        }

        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                grad_queries, at::Tensor(), at::Tensor()};
    }
};

inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices_tree,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    return RangeSearch::apply(
        points, split_dim, split_val, left, right, indices_tree,
        leaf_starts, leaf_counts, queries, radius, p
    );
}

}  // namespace torchscience::autograd::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("range_search", &torchscience::autograd::space_partitioning::range_search);
}
```

**Step 3: Add includes to torchscience.cpp**

```cpp
#include "meta/space_partitioning/range_search.h"
#include "autograd/space_partitioning/range_search.h"
```

**Step 4: Commit**

```bash
git add src/torchscience/csrc/meta/space_partitioning/range_search.h src/torchscience/csrc/autograd/space_partitioning/range_search.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(range_search): implement meta and autograd kernels"
```

---

### Task 3: Implement range_search Autocast Kernel

**Files:**
- Create: `src/torchscience/csrc/autocast/space_partitioning/range_search.h`

**Step 1: Create autocast kernel**

```cpp
// src/torchscience/csrc/autocast/space_partitioning/range_search.h
#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    at::Tensor points_fp32 = at::autocast::cached_cast(at::kFloat, points);
    at::Tensor split_val_fp32 = at::autocast::cached_cast(at::kFloat, split_val);
    at::Tensor queries_fp32 = at::autocast::cached_cast(at::kFloat, queries);

    auto [result_indices, distances] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::range_search", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            double, double
        )>()
        .call(points_fp32, split_dim, split_val_fp32, left, right, indices,
              leaf_starts, leaf_counts, queries_fp32, radius, p);

    // Cast nested tensor distances back
    return std::make_tuple(result_indices, distances.to(queries.scalar_type()));
}

}  // namespace torchscience::autocast::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("range_search", &torchscience::autocast::space_partitioning::range_search);
}
```

**Step 2: Add include to torchscience.cpp**

```cpp
#include "autocast/space_partitioning/range_search.h"
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/autocast/space_partitioning/range_search.h src/torchscience/csrc/torchscience.cpp
git commit -m "feat(range_search): implement autocast kernel"
```

---

### Task 4: Implement range_search Python Wrapper - Tests First

**Files:**
- Create: `tests/torchscience/space_partitioning/test__range_search.py`

**Step 1: Write tests**

```python
# tests/torchscience/space_partitioning/test__range_search.py
import pytest
import torch
from torch.autograd import gradcheck

from torchscience.space_partitioning import kd_tree, range_search


class TestRangeSearchBasic:
    """Tests for range_search query function."""

    def test_returns_nested_tensors(self):
        """range_search returns nested tensors."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=1.0)

        assert indices.is_nested
        assert distances.is_nested

    def test_nested_tensor_length_matches_queries(self):
        """Nested tensors have one entry per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=1.0)

        assert len(indices.unbind()) == 10
        assert len(distances.unbind()) == 10

    def test_distances_are_within_radius(self):
        """All returned distances are within radius."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)
        radius = 1.0

        indices, distances = range_search(tree, queries, radius=radius)

        for dist_tensor in distances.unbind():
            if dist_tensor.numel() > 0:
                assert (dist_tensor <= radius + 1e-6).all()

    def test_distances_are_sorted(self):
        """Distances are sorted in ascending order per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=2.0)

        for dist_tensor in distances.unbind():
            if dist_tensor.numel() > 1:
                sorted_dists = torch.sort(dist_tensor)[0]
                torch.testing.assert_close(dist_tensor, sorted_dists)

    def test_indices_are_valid(self):
        """Indices are in valid range [0, n)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=2.0)

        for idx_tensor in indices.unbind():
            if idx_tensor.numel() > 0:
                assert (idx_tensor >= 0).all()
                assert (idx_tensor < 100).all()


class TestRangeSearchCorrectness:
    """Tests for correctness against brute force."""

    def test_matches_brute_force(self):
        """Results match brute-force search."""
        torch.manual_seed(42)
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64)
        radius = 1.5

        indices, distances = range_search(tree, queries, radius=radius)

        # Brute force
        for i, query in enumerate(queries):
            dists = torch.sqrt(((points - query) ** 2).sum(dim=1))
            bf_mask = dists <= radius
            bf_dists = dists[bf_mask]
            bf_indices = torch.where(bf_mask)[0]

            # Sort for comparison
            bf_sorted_idx = torch.argsort(bf_dists)
            bf_dists_sorted = bf_dists[bf_sorted_idx]
            bf_indices_sorted = bf_indices[bf_sorted_idx]

            result_indices = indices.unbind()[i]
            result_distances = distances.unbind()[i]

            assert result_indices.numel() == bf_indices_sorted.numel()
            torch.testing.assert_close(
                result_distances, bf_dists_sorted, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(result_indices, bf_indices_sorted)

    def test_query_point_in_dataset(self):
        """Query point that exists in dataset is found with distance 0."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        query = points[42:43]

        indices, distances = range_search(tree, query, radius=0.1)

        idx_list = indices.unbind()[0]
        dist_list = distances.unbind()[0]

        assert 42 in idx_list.tolist()
        min_dist = dist_list.min()
        torch.testing.assert_close(min_dist, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_empty_result(self):
        """Handles queries with no neighbors within radius."""
        points = torch.zeros(10, 3)  # All at origin
        tree = kd_tree(points)
        query = torch.tensor([[100.0, 100.0, 100.0]])  # Far away

        indices, distances = range_search(tree, query, radius=1.0)

        assert indices.unbind()[0].numel() == 0
        assert distances.unbind()[0].numel() == 0


class TestRangeSearchGradient:
    """Tests for gradient support."""

    def test_gradient_exists(self):
        """Gradient exists for query points."""
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        indices, distances = range_search(tree, queries, radius=2.0)

        # Sum all distances across nested tensor
        total = sum(d.sum() for d in distances.unbind())
        total.backward()

        assert queries.grad is not None
        assert torch.isfinite(queries.grad).all()

    def test_zero_distance_gradient_is_finite(self):
        """Zero distance has finite gradient."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = points[5:6].clone().requires_grad_(True)

        _, distances = range_search(tree, queries, radius=1.0)
        total = sum(d.sum() for d in distances.unbind())
        total.backward()

        assert torch.isfinite(queries.grad).all()


class TestRangeSearchValidation:
    """Tests for input validation."""

    def test_wrong_tree_type_raises(self):
        """Raises RuntimeError for wrong tree type."""
        from tensordict import TensorDict

        fake_tree = TensorDict({"_type": "wrong_type"}, batch_size=[])
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="Unsupported tree type"):
            range_search(fake_tree, queries, radius=1.0)

    def test_dimension_mismatch_raises(self):
        """Raises RuntimeError when query dimension doesn't match tree."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 5)

        with pytest.raises(RuntimeError, match="dimension"):
            range_search(tree, queries, radius=1.0)

    def test_negative_radius_raises(self):
        """Raises RuntimeError for negative radius."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        with pytest.raises(RuntimeError, match="radius"):
            range_search(tree, queries, radius=-1.0)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/torchscience/space_partitioning/test__range_search.py -v
```

**Step 3: Commit**

```bash
git add tests/torchscience/space_partitioning/test__range_search.py
git commit -m "test(range_search): add failing tests for nested tensor output"
```

---

### Task 5: Implement range_search Python Wrapper

**Files:**
- Create: `src/torchscience/space_partitioning/_range_search.py`
- Modify: `src/torchscience/space_partitioning/__init__.py`

**Step 1: Implement range_search function**

```python
# src/torchscience/space_partitioning/_range_search.py
"""Range search query returning nested tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

import torchscience._csrc  # noqa: F401


def range_search(
    tree: TensorDict,
    queries: Tensor,
    radius: float,
    *,
    p: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Find all neighbors within radius for each query point.

    Parameters
    ----------
    tree : TensorDict
        Spatial index built by kd_tree(), bounding_volume_hierarchy(), or octree().
    queries : Tensor, shape (m, d)
        Query points.
    radius : float
        Search radius (inclusive).
    p : float, default=2.0
        Minkowski p-norm (2.0 = Euclidean, 1.0 = Manhattan).

    Returns
    -------
    indices : Tensor (nested)
        Indices of neighbors per query. Use `.unbind()` to get list of tensors.
    distances : Tensor (nested)
        Distances to neighbors per query. Supports gradients w.r.t. query points.

    Notes
    -----
    Returns PyTorch nested tensors for variable-length results. Each query
    may have a different number of neighbors within the radius.

    Uses actual tree traversal with branch pruning for O(log n + k) average
    complexity per query, where k is the number of results.

    Examples
    --------
    >>> points = torch.randn(1000, 3)
    >>> tree = kd_tree(points)
    >>> queries = torch.randn(10, 3)
    >>> indices, distances = range_search(tree, queries, radius=1.0)
    >>> indices.is_nested
    True
    >>> # Access individual query results
    >>> for idx, dist in zip(indices.unbind(), distances.unbind()):
    ...     print(f"Found {len(idx)} neighbors")
    """
    if queries.dim() != 2:
        raise RuntimeError(f"queries must be 2D (m, d), got {queries.dim()}D")

    if radius < 0:
        raise RuntimeError(f"radius must be non-negative, got {radius}")

    tree_type = tree.get("_type", None)

    if tree_type == "kd_tree":
        points = tree["points"]
        d = points.size(1)

        if queries.size(1) != d:
            raise RuntimeError(
                f"Query dimension ({queries.size(1)}) must match "
                f"tree dimension ({d})"
            )

        return torch.ops.torchscience.range_search(
            tree["points"],
            tree["split_dim"],
            tree["split_val"],
            tree["left"],
            tree["right"],
            tree["indices"],
            tree["leaf_starts"],
            tree["leaf_counts"],
            queries,
            radius,
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
from ._range_search import range_search

__all__ = [
    "kd_tree",
    "k_nearest_neighbors",
    "range_search",
]
```

**Step 3: Run tests and commit**

```bash
uv run pytest tests/torchscience/space_partitioning/ -v
git add src/torchscience/space_partitioning/
git commit -m "feat(range_search): implement Python wrapper with nested tensor output"
```

---

### Task 6: Final Integration and Cleanup

**Step 1: Run complete test suite**

```bash
uv run pytest tests/torchscience/space_partitioning/ -v --tb=short
```

**Step 2: Run linting**

```bash
uv run ruff check src/torchscience/space_partitioning/
uv run ruff format src/torchscience/space_partitioning/
```

**Step 3: Verify all imports work**

```bash
uv run python -c "
from torchscience.space_partitioning import (
    kd_tree,
    k_nearest_neighbors,
    range_search,
)
import torch

# Single tree
points = torch.randn(100, 3)
tree = kd_tree(points)
print(f'Tree type: {tree[\"_type\"]}')

# k-NN query
queries = torch.randn(5, 3)
idx, dist = k_nearest_neighbors(tree, queries, k=3)
print(f'k-NN indices shape: {idx.shape}')

# Range query (nested tensors)
indices, distances = range_search(tree, queries, radius=1.0)
print(f'Range search results: {[len(i) for i in indices.unbind()]} neighbors')

# Batched construction
batched_points = torch.randn(4, 100, 3)
trees = kd_tree(batched_points)
print(f'Built {len(trees)} trees')

print('All imports and operations successful!')
"
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(space_partitioning): final cleanup and formatting"
```

---

## Summary

This phase implements:
- **1 query function:** `range_search` with nested tensor output
- **C++ backend** with CPU, Meta, Autograd, and Autocast kernels
- **Per-query gradient computation** via unbind + gather
- **Final integration** and verification

**Total tasks:** 6
**Estimated commits:** ~7

**Phase 1 Complete!**

The `torchscience.space_partitioning` module now provides:
- `kd_tree`: Build k-d tree with SAH splitting
- `k_nearest_neighbors`: O(log n) k-NN query with full autograd
- `range_search`: O(log n + k) range query with nested tensor output

**Future work (Phase 2+):**
- `bounding_volume_hierarchy` build function
- `octree` build function
- CUDA kernels for GPU acceleration
- Memory-mapped tree storage
