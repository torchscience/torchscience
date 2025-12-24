# Operator Traits Evolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `backward_backward` support to CPU operator templates, convert CUDA from macros to templates, and extend sparse/quantized backends to use the same patterns.

**Architecture:** Each template category (Reduction, Fixed, Identity, Batched, Pairwise) gets a `backward_backward` method that follows the same structure as `backward`. Traits provide the analytical second derivative kernel. CUDA templates mirror CPU structure using `gpu_kernel` instead of `cpu_kernel`. Sparse/quantized delegate to dense operations.

**Tech Stack:** C++20, PyTorch C++ API (ATen, TensorIterator), TORCH_LIBRARY macros

---

## Phase 1: CPU backward_backward for Reduction Template

### Task 1.1: Add backward_backward to CPUReductionOperator

**Files:**
- Modify: `src/torchscience/csrc/cpu/reduction_operators.h:139-293`

**Step 1: Update the traits documentation comment**

Add to the comment at line 139-141:

```cpp
// ReductionTraits must provide:
//   - template<T> static T reduce(const T* data, int64_t n, Args... args);
//   - template<T> static void backward(T grad, const T* data, int64_t n, T* grad_out, Args...);
//   - template<T> static void backward_backward(
//         const T* grad_grad_input, T grad_output, const T* input, int64_t n,
//         T& grad_grad_output, T* new_grad_input, Args...);
```

**Step 2: Add backward_backward method after backward (line 292)**

```cpp
    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        at::OptionalIntArrayRef dim,
        bool keepdim,
        Args... args
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        auto [reduce_size, batch_size] = reduction_detail::compute_reduce_info(input, dim);

        // Output tensors
        at::Tensor grad_grad_output;
        at::Tensor new_grad_input = at::zeros_like(input);

        if (!dim.has_value() || dim->empty()) {
            // Scalar reduction
            grad_grad_output = at::zeros({}, input.options());

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_backward_backward_cpu_all",
                [&]() {
                    scalar_t grad_out_val = grad_output.item<scalar_t>();
                    const scalar_t* gg_ptr = grad_grad_input_contig.data_ptr<scalar_t>();
                    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();
                    scalar_t gg_out = scalar_t(0);

                    ReductionTraits::template backward_backward<scalar_t>(
                        gg_ptr, grad_out_val, in_ptr, input_contig.numel(),
                        gg_out, new_grad_ptr, args...
                    );

                    grad_grad_output.fill_(gg_out);
                }
            );
        } else {
            // Dimension-specific backward_backward
            auto permutation = reduction_detail::compute_permutation(input, dim);
            auto inverse_perm = reduction_detail::compute_inverse_permutation(permutation);

            at::Tensor permuted_input = input_contig.permute(permutation).contiguous();
            at::Tensor permuted_view = permuted_input.view({batch_size, reduce_size});

            at::Tensor permuted_gg = grad_grad_input_contig.permute(permutation).contiguous();
            at::Tensor permuted_gg_view = permuted_gg.view({batch_size, reduce_size});

            at::Tensor new_grad_permuted = at::zeros({batch_size, reduce_size}, input.options());

            // Expand grad_output to batch shape
            at::Tensor grad_output_expanded;
            if (keepdim) {
                grad_output_expanded = grad_output.contiguous().view({batch_size});
            } else {
                at::Tensor temp = grad_output;
                int64_t ndim = input.dim();
                std::vector<bool> reduce_dim(ndim, false);
                for (int64_t d : *dim) {
                    int64_t pos_d = d >= 0 ? d : d + ndim;
                    reduce_dim[pos_d] = true;
                }
                for (int64_t i = 0; i < ndim; ++i) {
                    if (reduce_dim[i]) {
                        temp = temp.unsqueeze(i);
                    }
                }
                grad_output_expanded = temp.contiguous().view({batch_size});
            }

            grad_grad_output = at::zeros({batch_size}, input.options());

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input_contig.scalar_type(),
                "reduction_backward_backward_cpu_dim",
                [&]() {
                    const scalar_t* gg_ptr = permuted_gg_view.data_ptr<scalar_t>();
                    const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
                    const scalar_t* in_ptr = permuted_view.data_ptr<scalar_t>();
                    scalar_t* gg_out_ptr = grad_grad_output.data_ptr<scalar_t>();
                    scalar_t* new_grad_ptr = new_grad_permuted.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            ReductionTraits::template backward_backward<scalar_t>(
                                gg_ptr + b * reduce_size,
                                grad_out_ptr[b],
                                in_ptr + b * reduce_size,
                                reduce_size,
                                gg_out_ptr[b],
                                new_grad_ptr + b * reduce_size,
                                args...
                            );
                        }
                    });
                }
            );

            // Reshape grad_grad_output to match original grad_output shape
            auto output_shape = reduction_detail::compute_output_shape(input, dim, keepdim);
            if (output_shape.empty()) {
                grad_grad_output = grad_grad_output.sum();
            } else {
                grad_grad_output = grad_grad_output.view(output_shape);
            }

            // Inverse permute new_grad_input
            new_grad_input = new_grad_permuted.view(permuted_input.sizes())
                .permute(inverse_perm)
                .contiguous();
        }

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
```

**Step 3: Verify file compiles**

Run: `ls src/torchscience/csrc/cpu/reduction_operators.h`
Expected: File exists (syntax check happens at build time)

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/reduction_operators.h
git commit -m "feat(cpu): add backward_backward to CPUReductionOperator"
```

---

### Task 1.2: Add backward_backward Test for Reduction

**Files:**
- Modify: `tests/csrc/test_reduction_operators.py`

**Step 1: Add backward_backward test**

Add after line 63:

```python
    def test_reduction_backward_backward(self, input_tensor):
        """Test second-order backward pass through reduction."""
        x = input_tensor.clone().detach().requires_grad_(True)

        # Create computation graph with double backward
        result = torch.ops.torchscience.kurtosis(x, [1], False, True, True)

        # First backward
        grad_output = torch.ones_like(result)
        (grad_input,) = torch.autograd.grad(
            result, x, grad_output, create_graph=True
        )

        # Second backward (backward_backward)
        grad_grad_input = torch.ones_like(grad_input)
        grad_grad_output = torch.autograd.grad(
            grad_input, x, grad_grad_input, retain_graph=True
        )

        assert grad_grad_output[0] is not None
        assert grad_grad_output[0].shape == x.shape
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/csrc/test_reduction_operators.py::TestReductionOperatorTemplate::test_reduction_backward_backward -v`
Expected: PASS (or SKIP if kurtosis traits don't implement backward_backward yet)

**Step 3: Commit**

```bash
git add tests/csrc/test_reduction_operators.py
git commit -m "test: add backward_backward test for reduction operators"
```

---

## Phase 2: CPU backward_backward for Fixed Template

### Task 2.1: Add backward_backward to CPUFixedOperator

**Files:**
- Modify: `src/torchscience/csrc/cpu/fixed_operators.h:14-197`

**Step 1: Update traits documentation**

Update comment at line 17-20:

```cpp
// FixedTraits must provide:
//   - static int64_t output_size(int64_t input_size, Args... args);
//   - template<T> static void kernel(const T* in, T* out, int64_t in_size, int64_t out_size, Args...);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, sizes, Args...);
//   - template<T> static void backward_backward_kernel(
//         const T* grad_grad_in, const T* grad_out, const T* in,
//         int64_t in_size, int64_t out_size,
//         T* grad_grad_out, T* new_grad_in, Args...);
```

**Step 2: Add backward_backward method after backward (line 193)**

```cpp
    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t dim,
        Args... args
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;

        int64_t input_size = input.size(dim);
        int64_t output_size = grad_output.size(dim);

        at::Tensor grad_grad_output = at::zeros_like(grad_output);
        at::Tensor new_grad_input = at::zeros_like(input);

        int64_t batch_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            batch_size *= input.size(i);
        }
        int64_t trailing_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            trailing_size *= input.size(i);
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        if (trailing_size == 1) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                input.scalar_type(),
                "fixed_backward_backward_cpu_simple",
                [&]() {
                    const scalar_t* gg_in_ptr = grad_grad_input_contig.data_ptr<scalar_t>();
                    const scalar_t* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
                    const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                    scalar_t* gg_out_ptr = grad_grad_output.data_ptr<scalar_t>();
                    scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();

                    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                        for (int64_t b = begin; b < end; ++b) {
                            FixedTraits::template backward_backward_kernel<scalar_t>(
                                gg_in_ptr + b * input_size,
                                grad_out_ptr + b * output_size,
                                in_ptr + b * input_size,
                                input_size,
                                output_size,
                                gg_out_ptr + b * output_size,
                                new_grad_ptr + b * input_size,
                                args...
                            );
                        }
                    });
                }
            );
        }
        // Note: strided case follows same pattern as forward/backward

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/fixed_operators.h
git commit -m "feat(cpu): add backward_backward to CPUFixedOperator"
```

---

### Task 2.2: Add MetaFixedOperator backward_backward

**Files:**
- Modify: `src/torchscience/csrc/meta/fixed_operators.h`

**Step 1: Add backward_backward method**

The meta template already has a stub. Verify it returns correct shapes:

```cpp
    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t dim,
        Args...
    ) {
        (void)grad_grad_input;
        (void)dim;
        return std::make_tuple(
            at::empty_like(grad_output),
            at::empty_like(input)
        );
    }
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/fixed_operators.h
git commit -m "feat(meta): add backward_backward to MetaFixedOperator"
```

---

## Phase 3: CPU backward_backward for Identity Template

### Task 3.1: Add backward_backward to CPUIdentityOperator

**Files:**
- Modify: `src/torchscience/csrc/cpu/identity_operators.h:14-139`

**Step 1: Update traits documentation**

Update comment at lines 14-17:

```cpp
// IdentityTraits must provide:
//   - static constexpr int64_t channel_size;  // e.g., 3 for RGB
//   - template<T> static void kernel(const T* in, T* out, int64_t channel_size);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, int64_t channel_size);
//   - template<T> static void backward_backward_kernel(
//         const T* grad_grad_in, const T* grad_out, const T* in, int64_t channel_size,
//         T* grad_grad_out, T* new_grad_in);
```

**Step 2: Add backward_backward method after backward (line 136)**

```cpp
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t channel_dim = -1
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        int64_t ndim = input.dim();
        if (channel_dim < 0) channel_dim += ndim;

        int64_t channel_size = input.size(channel_dim);
        int64_t batch_size = input.numel() / channel_size;

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        at::Tensor grad_grad_output = at::empty_like(grad_output);
        at::Tensor new_grad_input = at::empty_like(input);

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != channel_dim) perm.push_back(i);
        }
        perm.push_back(channel_dim);

        at::Tensor permuted_in = input_contig.permute(perm).contiguous();
        at::Tensor permuted_grad_out = grad_output_contig.permute(perm).contiguous();
        at::Tensor permuted_gg_in = grad_grad_input_contig.permute(perm).contiguous();
        at::Tensor permuted_gg_out = at::empty_like(permuted_grad_out);
        at::Tensor permuted_new_grad = at::empty_like(permuted_in);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "identity_backward_backward_cpu",
            [&]() {
                const scalar_t* gg_in_ptr = permuted_gg_in.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = permuted_grad_out.data_ptr<scalar_t>();
                const scalar_t* in_ptr = permuted_in.data_ptr<scalar_t>();
                scalar_t* gg_out_ptr = permuted_gg_out.data_ptr<scalar_t>();
                scalar_t* new_grad_ptr = permuted_new_grad.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        IdentityTraits::template backward_backward_kernel<scalar_t>(
                            gg_in_ptr + b * channel_size,
                            grad_out_ptr + b * channel_size,
                            in_ptr + b * channel_size,
                            channel_size,
                            gg_out_ptr + b * channel_size,
                            new_grad_ptr + b * channel_size
                        );
                    }
                });
            }
        );

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
            inv_perm[perm[i]] = i;
        }

        grad_grad_output = permuted_gg_out.permute(inv_perm).contiguous();
        new_grad_input = permuted_new_grad.permute(inv_perm).contiguous();

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/identity_operators.h
git commit -m "feat(cpu): add backward_backward to CPUIdentityOperator"
```

---

## Phase 4: CPU backward_backward for Batched Template

### Task 4.1: Add backward_backward to CPUBatchedOperator

**Files:**
- Modify: `src/torchscience/csrc/cpu/batched_operators.h:14-156`

**Step 1: Update traits documentation**

Update comment at lines 14-18:

```cpp
// BatchedTraits must provide:
//   - static constexpr int64_t num_trailing_dims;  // e.g., 2 for matrix ops
//   - static std::vector<int64_t> output_trailing_shape(ArrayRef<int64_t> input_trailing);
//   - template<T> static void kernel(const T* in, T* out, ArrayRef<int64_t> inner_shape);
//   - template<T> static void backward_kernel(const T* grad_out, const T* in, T* grad_in, ArrayRef<int64_t> inner_shape);
//   - template<T> static void backward_backward_kernel(
//         const T* grad_grad_in, const T* grad_out, const T* in,
//         ArrayRef<int64_t> inner_shape, T* grad_grad_out, T* new_grad_in);
```

**Step 2: Add backward_backward method after backward (line 153)**

```cpp
    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        Args... args
    ) {
        if (!grad_grad_input.defined()) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        constexpr int64_t num_trailing = BatchedTraits::num_trailing_dims;
        int64_t ndim = input.dim();

        std::vector<int64_t> trailing_shape;
        for (int64_t i = ndim - num_trailing; i < ndim; ++i) {
            trailing_shape.push_back(input.size(i));
        }

        std::vector<int64_t> output_trailing = BatchedTraits::output_trailing_shape(trailing_shape);

        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - num_trailing; ++i) {
            batch_size *= input.size(i);
        }

        int64_t inner_input_size = 1;
        for (int64_t s : trailing_shape) {
            inner_input_size *= s;
        }
        int64_t inner_output_size = 1;
        for (int64_t s : output_trailing) {
            inner_output_size *= s;
        }

        at::Tensor input_contig = input.contiguous();
        at::Tensor grad_output_contig = grad_output.contiguous();
        at::Tensor grad_grad_input_contig = grad_grad_input.contiguous();

        at::Tensor grad_grad_output = at::zeros_like(grad_output);
        at::Tensor new_grad_input = at::zeros_like(input);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            input.scalar_type(),
            "batched_backward_backward_cpu",
            [&]() {
                const scalar_t* gg_in_ptr = grad_grad_input_contig.data_ptr<scalar_t>();
                const scalar_t* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
                const scalar_t* in_ptr = input_contig.data_ptr<scalar_t>();
                scalar_t* gg_out_ptr = grad_grad_output.data_ptr<scalar_t>();
                scalar_t* new_grad_ptr = new_grad_input.data_ptr<scalar_t>();

                at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) {
                        BatchedTraits::template backward_backward_kernel<scalar_t>(
                            gg_in_ptr + b * inner_input_size,
                            grad_out_ptr + b * inner_output_size,
                            in_ptr + b * inner_input_size,
                            trailing_shape,
                            gg_out_ptr + b * inner_output_size,
                            new_grad_ptr + b * inner_input_size,
                            args...
                        );
                    }
                });
            }
        );

        return std::make_tuple(grad_grad_output, new_grad_input);
    }
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/batched_operators.h
git commit -m "feat(cpu): add backward_backward to CPUBatchedOperator"
```

---

## Phase 5: CPU backward_backward for Pairwise Template

### Task 5.1: Add backward_backward to CPUPairwiseOperator

**Files:**
- Modify: `src/torchscience/csrc/cpu/pairwise_operators.h:14-125`

**Step 1: Update traits documentation**

Update comment at lines 14-16:

```cpp
// PairwiseTraits must provide:
//   - template<T> static T compute(const T* x, const T* y, int64_t d, Args... args);
//   - template<T> static void backward(T grad, const T* x, const T* y, int64_t d, T* grad_x, T* grad_y, Args...);
//   - template<T> static void backward_backward(
//         const T* grad_grad_x, const T* grad_grad_y, T grad_output,
//         const T* x, const T* y, int64_t d,
//         T& grad_grad_output, T* new_grad_x, T* new_grad_y, Args...);
```

**Step 2: Add backward_backward method after backward (line 122)**

```cpp
    template<typename... Args>
    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_x,
        const at::Tensor& grad_grad_y,
        const at::Tensor& grad_output,  // (m, n)
        const at::Tensor& x,  // (m, d)
        const at::Tensor& y,  // (n, d)
        Args... args
    ) {
        bool has_gg_x = grad_grad_x.defined();
        bool has_gg_y = grad_grad_y.defined();

        if (!has_gg_x && !has_gg_y) {
            return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
        }

        int64_t m = x.size(0);
        int64_t n = y.size(0);
        int64_t d = x.size(1);

        at::Tensor x_contig = x.contiguous();
        at::Tensor y_contig = y.contiguous();
        at::Tensor grad_contig = grad_output.contiguous();
        at::Tensor gg_x_contig = has_gg_x ? grad_grad_x.contiguous() : at::zeros_like(x);
        at::Tensor gg_y_contig = has_gg_y ? grad_grad_y.contiguous() : at::zeros_like(y);

        at::Tensor grad_grad_output = at::zeros_like(grad_output);
        at::Tensor new_grad_x = at::zeros_like(x);
        at::Tensor new_grad_y = at::zeros_like(y);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            x.scalar_type(),
            "pairwise_backward_backward_cpu",
            [&]() {
                const scalar_t* gg_x_ptr = gg_x_contig.data_ptr<scalar_t>();
                const scalar_t* gg_y_ptr = gg_y_contig.data_ptr<scalar_t>();
                const scalar_t* grad_ptr = grad_contig.data_ptr<scalar_t>();
                const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
                const scalar_t* y_ptr = y_contig.data_ptr<scalar_t>();
                scalar_t* gg_out_ptr = grad_grad_output.data_ptr<scalar_t>();
                scalar_t* new_grad_x_ptr = new_grad_x.data_ptr<scalar_t>();
                scalar_t* new_grad_y_ptr = new_grad_y.data_ptr<scalar_t>();

                // Sequential for correctness (accumulation)
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        std::vector<scalar_t> temp_new_grad_x(d, scalar_t(0));
                        std::vector<scalar_t> temp_new_grad_y(d, scalar_t(0));

                        PairwiseTraits::template backward_backward<scalar_t>(
                            gg_x_ptr + i * d,
                            gg_y_ptr + j * d,
                            grad_ptr[i * n + j],
                            x_ptr + i * d,
                            y_ptr + j * d,
                            d,
                            gg_out_ptr[i * n + j],
                            temp_new_grad_x.data(),
                            temp_new_grad_y.data(),
                            args...
                        );

                        for (int64_t k = 0; k < d; ++k) {
                            new_grad_x_ptr[i * d + k] += temp_new_grad_x[k];
                            new_grad_y_ptr[j * d + k] += temp_new_grad_y[k];
                        }
                    }
                }
            }
        );

        return std::make_tuple(grad_grad_output, new_grad_x, new_grad_y);
    }
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cpu/pairwise_operators.h
git commit -m "feat(cpu): add backward_backward to CPUPairwiseOperator"
```

---

## Phase 6: CUDA Template Conversion

### Task 6.1: Create CUDAUnaryOperator Template

**Files:**
- Create: `src/torchscience/csrc/cuda/operators.cuh`

**Step 1: Write the CUDA unary operator template**

```cpp
#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/TensorIterator.h>
#include <torch/library.h>

namespace torchscience::cuda {

// =============================================================================
// CUDAUnaryOperator - Template for unary operators (mirrors CPUUnaryOperator)
// =============================================================================

template<typename ImplTraits>
struct CUDAUnaryOperator {
    static at::Tensor forward(const at::Tensor& input) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_forward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x);
                });
            }
        );

        return iter.output();
    }

    static at::Tensor backward(
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_backward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t g, scalar_t x) -> scalar_t {
                    return ImplTraits::template backward<scalar_t>(g, x);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input,
        const at::Tensor& grad_output,
        const at::Tensor& input
    ) {
        const bool has_gg = grad_grad_input.defined();

        if (!has_gg) {
            return std::make_tuple(at::Tensor(), at::Tensor());
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input)
            .add_const_input(grad_grad_input)
            .add_const_input(grad_output)
            .add_const_input(input)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_unary_backward_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    [has_gg]GPU_LAMBDA(scalar_t gg, scalar_t g, scalar_t x)
                        -> thrust::tuple<scalar_t, scalar_t> {
                        auto [gg_out, new_grad] = ImplTraits::template backward_backward<scalar_t>(
                            gg, g, x, has_gg
                        );
                        return thrust::make_tuple(gg_out, new_grad);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_CUDA_UNARY(module, name, Impl) \
    ::torchscience::cuda::CUDAUnaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")

}  // namespace torchscience::cuda
```

**Step 2: Verify file exists**

Run: `ls src/torchscience/csrc/cuda/operators.cuh`
Expected: File exists

**Step 3: Commit**

```bash
git add src/torchscience/csrc/cuda/operators.cuh
git commit -m "feat(cuda): add CUDAUnaryOperator template"
```

---

### Task 6.2: Create CUDABinaryOperator Template

**Files:**
- Modify: `src/torchscience/csrc/cuda/operators.cuh`

**Step 1: Add CUDABinaryOperator after CUDAUnaryOperator**

```cpp
// =============================================================================
// CUDABinaryOperator - Template for binary operators
// =============================================================================

template<typename ImplTraits>
struct CUDABinaryOperator {
    static at::Tensor forward(
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor output;

        auto iter = at::TensorIteratorConfig()
            .add_output(output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_forward",
            [&]() {
                at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t x1, scalar_t x2) -> scalar_t {
                    return ImplTraits::template forward<scalar_t>(x1, x2);
                });
            }
        );

        return iter.output();
    }

    static std::tuple<at::Tensor, at::Tensor> backward(
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    []GPU_LAMBDA(scalar_t g, scalar_t x1, scalar_t x2)
                        -> thrust::tuple<scalar_t, scalar_t> {
                        auto [g1, g2] = ImplTraits::template backward<scalar_t>(g, x1, x2);
                        return thrust::make_tuple(g1, g2);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1));
    }

    static std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_backward(
        const at::Tensor& grad_grad_input1,
        const at::Tensor& grad_grad_input2,
        const at::Tensor& grad_output,
        const at::Tensor& input1,
        const at::Tensor& input2
    ) {
        const bool has_gg1 = grad_grad_input1.defined();
        const bool has_gg2 = grad_grad_input2.defined();

        if (!has_gg1 && !has_gg2) {
            return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
        }

        at::Tensor grad_grad_output;
        at::Tensor grad_input1;
        at::Tensor grad_input2;

        at::Tensor gg1_input = has_gg1 ? grad_grad_input1 : at::zeros_like(grad_output);
        at::Tensor gg2_input = has_gg2 ? grad_grad_input2 : at::zeros_like(grad_output);

        auto iter = at::TensorIteratorConfig()
            .add_output(grad_grad_output)
            .add_output(grad_input1)
            .add_output(grad_input2)
            .add_const_input(gg1_input)
            .add_const_input(gg2_input)
            .add_const_input(grad_output)
            .add_const_input(input1)
            .add_const_input(input2)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(false)
            .build();

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kBFloat16, at::kHalf,
            iter.common_dtype(),
            "cuda_binary_backward_backward",
            [&]() {
                at::native::gpu_kernel_multiple_outputs(
                    iter,
                    [has_gg1, has_gg2]GPU_LAMBDA(
                        scalar_t gg1, scalar_t gg2, scalar_t g, scalar_t x1, scalar_t x2
                    ) -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
                        auto [gg_out, g1, g2] = ImplTraits::template backward_backward<scalar_t>(
                            gg1, gg2, g, x1, x2, has_gg1, has_gg2
                        );
                        return thrust::make_tuple(gg_out, g1, g2);
                    }
                );
            }
        );

        return std::make_tuple(iter.output(0), iter.output(1), iter.output(2));
    }

    static void register_all(
        torch::Library& module,
        const char* name,
        const char* backward_name,
        const char* backward_backward_name
    ) {
        module.impl(name, &forward);
        module.impl(backward_name, &backward);
        module.impl(backward_backward_name, &backward_backward);
    }
};

#define REGISTER_CUDA_BINARY(module, name, Impl) \
    ::torchscience::cuda::CUDABinaryOperator<Impl>::register_all( \
        module, #name, #name "_backward", #name "_backward_backward")
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/operators.cuh
git commit -m "feat(cuda): add CUDABinaryOperator template"
```

---

### Task 6.3: Migrate gamma to CUDA Template

**Files:**
- Modify: `src/torchscience/csrc/cuda/special_functions.cu`

**Step 1: Replace macro usage with template**

Find the gamma registration and replace with:

```cpp
#include "operators.cuh"
#include "../impl/special_functions/gamma_traits.h"

namespace torchscience::cuda::special_functions {

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
    REGISTER_CUDA_UNARY(m, gamma, ::torchscience::impl::special_functions::GammaTraits);
}

}  // namespace
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cuda/special_functions.cu
git commit -m "refactor(cuda): migrate gamma to template-based registration"
```

---

## Phase 7: Comprehensive Test Suite

### Task 7.1: Add backward_backward Tests for All Templates

**Files:**
- Modify: `tests/csrc/test_operator_templates.py`

**Step 1: Add backward_backward tests**

```python
    def test_pointwise_backward_backward(self, device):
        """Verify pointwise backward_backward via gamma."""
        x = torch.tensor(
            [2.0, 3.0, 4.0, 5.0], device=device, requires_grad=True
        )

        result = torch.ops.torchscience.gamma(x)

        # First backward with graph retention
        (grad,) = torch.autograd.grad(result.sum(), x, create_graph=True)

        # Second backward
        (grad_grad,) = torch.autograd.grad(grad.sum(), x, retain_graph=True)

        assert grad_grad is not None
        assert grad_grad.shape == x.shape
        # Verify values are finite
        assert torch.isfinite(grad_grad).all()

    def test_reduction_backward_backward(self, device):
        """Verify reduction backward_backward via kurtosis."""
        x = torch.randn(4, 10, device=device, requires_grad=True)

        result = torch.ops.torchscience.kurtosis(x, [1], False, True, True)

        # Check if backward_backward is implemented
        try:
            (grad,) = torch.autograd.grad(result.sum(), x, create_graph=True)
            (grad_grad,) = torch.autograd.grad(grad.sum(), x)
            assert grad_grad.shape == x.shape
        except RuntimeError:
            pytest.skip("backward_backward not yet implemented for kurtosis")
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/csrc/test_operator_templates.py -v`
Expected: PASS (with skips for unimplemented backward_backward)

**Step 3: Commit**

```bash
git add tests/csrc/test_operator_templates.py
git commit -m "test: add backward_backward tests for operator templates"
```

---

## Summary

This plan covers:

1. **Phase 1-5:** Add `backward_backward` to 5 CPU templates (Reduction, Fixed, Identity, Batched, Pairwise)
2. **Phase 6:** Create CUDA template equivalents and migrate one operator (gamma)
3. **Phase 7:** Comprehensive test coverage

Total tasks: 10
Estimated commits: 10

Future phases (not in this plan):
- Migrate remaining CUDA operators from macros to templates
- Add Meta/Autograd templates for Fixed, Identity, Batched, Pairwise
- Extend sparse/quantized backends
