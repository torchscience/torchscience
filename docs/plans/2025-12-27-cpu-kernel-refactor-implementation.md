# CPU Special Function Kernel Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor CPU special function operators to separate pure scalar kernels from TensorIterator dispatch.

**Architecture:** Each operator has exactly three scalar kernel functions (`*_forward_kernel`, `*_backward_kernel`, `*_backward_backward_kernel`) with all math inlined. TensorIterator dispatch functions become thin wrappers that pass the kernel directly.

**Tech Stack:** C++, ATen TensorIterator, PyTorch dispatcher

**Scope:** Only `gamma` and `chebyshev_polynomial_t` (the other two are stubs).

---

## Task 1: Refactor gamma.h - Extract forward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/gamma.h`

**Step 1: Replace gamma_kernel with gamma_forward_kernel**

Replace the current `gamma_kernel` function with `gamma_forward_kernel`:

```cpp
template <typename T> T gamma_forward_kernel(T z) {
  constexpr double kGammaG = 7.0;
  constexpr double kGammaCoefficients[] = {
      0.99999999999980993,  676.5203681218851,
      -1259.1392167224028,  771.32342877765313,
      -176.61502916214059,  12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7};

  if (z < T(0.5)) {
    return static_cast<T>(M_PI) / (std::sin(static_cast<T>(M_PI) * z) * gamma_forward_kernel(T(1) - z));
  }

  T z_adj = z - T(1);
  T x = static_cast<T>(kGammaCoefficients[0]);
  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(kGammaCoefficients[i]) / (z_adj + T(i));
  }

  const T g = static_cast<T>(kGammaG);
  T t = z_adj + g + T(0.5);
  return std::sqrt(static_cast<T>(2 * M_PI)) * std::pow(t, z_adj + T(0.5)) * std::exp(-t) * x;
}
```

**Step 2: Update gamma_forward to pass kernel directly**

Update the dispatch call:

```cpp
inline at::Tensor gamma_forward(const at::Tensor &input) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_cpu",
    [&] {
      at::native::cpu_kernel(iterator, gamma_forward_kernel<scalar_t>);
    }
  );

  return iterator.output();
}
```

**Step 3: Remove old constants outside anonymous namespace**

Delete these lines at the top (they'll be inside the kernel now):
```cpp
inline constexpr double kGammaG = 7.0;
inline constexpr double kGammaCoefficients[] = {...};
```

**Step 4: Run tests to verify forward still works**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__gamma.py -v -k "forward or test_basic" 2>&1 | head -50`

Expected: Tests pass

**Step 5: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/gamma.h
git commit -m "refactor(gamma): extract gamma_forward_kernel with inlined constants"
```

---

## Task 2: Refactor gamma.h - Extract backward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/gamma.h`

**Step 1: Create gamma_backward_kernel with inlined digamma**

Add this kernel (replace digamma_kernel usage):

```cpp
template <typename T> T gamma_backward_kernel(T g, T z) {
  // Compute gamma(z)
  T gamma_z = gamma_forward_kernel(z);

  // Compute digamma(z) - inlined
  T psi = T(0);
  T x = z;
  while (x < T(6)) {
    psi -= T(1) / x;
    x += T(1);
  }
  T x2 = T(1) / (x * x);
  psi += std::log(x) - T(0.5) / x -
         x2 * (T(1.0 / 12) - x2 * (T(1.0 / 120) - x2 * T(1.0 / 252)));

  return g * gamma_z * psi;
}
```

**Step 2: Update gamma_backward to pass kernel directly**

```cpp
inline at::Tensor gamma_backward(
  const at::Tensor &grad,
  const at::Tensor &input
) {
  at::Tensor grad_input;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(grad)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_backward_cpu",
    [&] {
      at::native::cpu_kernel(iterator, gamma_backward_kernel<scalar_t>);
    }
  );

  return iterator.output();
}
```

**Step 3: Delete digamma_kernel**

Remove the old `digamma_kernel` function entirely.

**Step 4: Run tests to verify backward still works**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__gamma.py -v 2>&1 | head -50`

Expected: Tests pass

**Step 5: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/gamma.h
git commit -m "refactor(gamma): extract gamma_backward_kernel with inlined digamma"
```

---

## Task 3: Refactor gamma.h - Extract backward_backward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/gamma.h`

**Step 1: Create gamma_backward_backward_kernel with inlined trigamma**

```cpp
template <typename T>
std::tuple<T, T> gamma_backward_backward_kernel(T gg, T g, T z) {
  // Compute gamma(z)
  T gamma_z = gamma_forward_kernel(z);

  // Compute digamma(z) - inlined
  T psi = T(0);
  T x = z;
  while (x < T(6)) {
    psi -= T(1) / x;
    x += T(1);
  }
  T x2 = T(1) / (x * x);
  psi += std::log(x) - T(0.5) / x -
         x2 * (T(1.0 / 12) - x2 * (T(1.0 / 120) - x2 * T(1.0 / 252)));

  // Compute trigamma(z) - inlined
  T psi1 = T(0);
  T y = z;
  while (y < T(6)) {
    psi1 += T(1) / (y * y);
    y += T(1);
  }
  T y2 = T(1) / (y * y);
  psi1 += T(1) / y + T(0.5) * y2 +
          y2 / y * (T(1.0 / 6) - y2 * (T(1.0 / 30) - y2 * T(1.0 / 42)));

  T gg_out = gg * gamma_z * psi;
  T new_grad = gg * g * gamma_z * (psi * psi + psi1);

  return {gg_out, new_grad};
}
```

**Step 2: Update gamma_backward_backward to pass kernel directly**

```cpp
inline std::tuple<at::Tensor, at::Tensor> gamma_backward_backward(
  const at::Tensor &gg_input,
  const at::Tensor &grad,
  const at::Tensor &input
) {
  if (!gg_input.defined()) {
    return {at::Tensor(), at::Tensor()};
  }

  at::Tensor gg_output;
  at::Tensor new_grad;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_output)
    .add_output(new_grad)
    .add_const_input(gg_input)
    .add_const_input(grad)
    .add_const_input(input)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "gamma_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        gamma_backward_backward_kernel<scalar_t>
      );
    }
  );

  return {iterator.output(0), iterator.output(1)};
}
```

**Step 3: Delete trigamma_kernel**

Remove the old `trigamma_kernel` function entirely.

**Step 4: Run full gamma test suite**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__gamma.py -v 2>&1`

Expected: All gamma tests pass

**Step 5: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/gamma.h
git commit -m "refactor(gamma): extract gamma_backward_backward_kernel with inlined trigamma"
```

---

## Task 4: Refactor chebyshev_polynomial_t.h - Extract forward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h`

**Step 1: Rename chebyshev_polynomial_t_kernel to chebyshev_polynomial_t_forward_kernel**

The existing kernel is already well-structured. Just rename it:

```cpp
template <typename T> T chebyshev_polynomial_t_forward_kernel(T x, T n) {
  if (std::abs(x) <= T(1)) {
    return std::cos(n * std::acos(x));
  }
  if (x > T(1)) {
    return std::cosh(n * std::acosh(x));
  }
  T sign = (static_cast<int>(n) % 2 == 0) ? T(1) : T(-1);
  return sign * std::cosh(n * std::acosh(-x));
}
```

**Step 2: Update chebyshev_polynomial_t_forward to pass kernel directly**

```cpp
inline at::Tensor chebyshev_polynomial_t_forward(
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor output;

  auto iterator = at::TensorIteratorConfig()
    .add_output(output)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_cpu",
    [&] {
      at::native::cpu_kernel(iterator, chebyshev_polynomial_t_forward_kernel<scalar_t>);
    }
  );

  return iterator.output();
}
```

**Step 3: Run tests to verify forward still works**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__chebyshev_polynomial_t.py -v -k "forward or test_basic" 2>&1 | head -50`

Expected: Tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h
git commit -m "refactor(chebyshev): rename to chebyshev_polynomial_t_forward_kernel"
```

---

## Task 5: Refactor chebyshev_polynomial_t.h - Extract backward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h`

**Step 1: Create chebyshev_polynomial_t_backward_kernel**

```cpp
template <typename T>
std::tuple<T, T> chebyshev_polynomial_t_backward_kernel(T g, T x, T n) {
  T eps = T(1e-6);
  T grad_x = g * n * (chebyshev_polynomial_t_forward_kernel(x + eps, n) -
                       chebyshev_polynomial_t_forward_kernel(x - eps, n)) / (T(2) * eps);
  return {grad_x, T(0)};
}
```

**Step 2: Update chebyshev_polynomial_t_backward to pass kernel directly**

```cpp
inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_t_backward(
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  at::Tensor grad_x;
  at::Tensor grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(grad_x)
    .add_output(grad_n)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        chebyshev_polynomial_t_backward_kernel<scalar_t>
      );
    }
  );

  return {iterator.output(0), iterator.output(1)};
}
```

**Step 3: Run tests to verify backward still works**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__chebyshev_polynomial_t.py -v 2>&1 | head -50`

Expected: Tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h
git commit -m "refactor(chebyshev): extract chebyshev_polynomial_t_backward_kernel"
```

---

## Task 6: Refactor chebyshev_polynomial_t.h - Extract backward_backward kernel

**Files:**
- Modify: `src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h`

**Step 1: Create chebyshev_polynomial_t_backward_backward_kernel**

```cpp
template <typename T>
std::tuple<T, T, T> chebyshev_polynomial_t_backward_backward_kernel(
  T gg_x, T gg_n, T g, T x, T n, bool has_gg_x
) {
  T eps = T(1e-5);
  T d2 = (chebyshev_polynomial_t_forward_kernel(x + eps, n) -
          T(2) * chebyshev_polynomial_t_forward_kernel(x, n) +
          chebyshev_polynomial_t_forward_kernel(x - eps, n)) / (eps * eps);

  T gg_out = has_gg_x ? gg_x * n * (chebyshev_polynomial_t_forward_kernel(x + eps, n) -
                                     chebyshev_polynomial_t_forward_kernel(x - eps, n)) / (T(2) * eps)
                      : T(0);
  T new_grad_x = has_gg_x ? gg_x * g * n * n * d2 : T(0);

  return {gg_out, new_grad_x, T(0)};
}
```

**Step 2: Update chebyshev_polynomial_t_backward_backward**

Note: This one is tricky because the kernel needs `has_gg_x` context. We'll use a lambda wrapper:

```cpp
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chebyshev_polynomial_t_backward_backward(
  const at::Tensor &gg_x,
  const at::Tensor &gg_n,
  const at::Tensor &grad,
  const at::Tensor &x,
  const at::Tensor &n
) {
  bool has_gg_x = gg_x.defined();
  bool has_gg_n = gg_n.defined();

  if (!has_gg_x && !has_gg_n) {
    return {at::Tensor(), at::Tensor(), at::Tensor()};
  }

  at::Tensor gg_x_safe = has_gg_x ? gg_x : at::zeros_like(grad);
  at::Tensor gg_n_safe = has_gg_n ? gg_n : at::zeros_like(grad);

  at::Tensor gg_out;
  at::Tensor new_grad_x;
  at::Tensor new_grad_n;

  auto iterator = at::TensorIteratorConfig()
    .add_output(gg_out)
    .add_output(new_grad_x)
    .add_output(new_grad_n)
    .add_const_input(gg_x_safe)
    .add_const_input(gg_n_safe)
    .add_const_input(grad)
    .add_const_input(x)
    .add_const_input(n)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    iterator.common_dtype(),
    "chebyshev_polynomial_t_backward_backward_cpu",
    [&] {
      at::native::cpu_kernel_multiple_outputs(
        iterator,
        [has_gg_x](scalar_t gg_x, scalar_t gg_n, scalar_t g, scalar_t x, scalar_t n) {
          return chebyshev_polynomial_t_backward_backward_kernel<scalar_t>(
            gg_x, gg_n, g, x, n, has_gg_x
          );
        }
      );
    }
  );

  return {iterator.output(0), iterator.output(1), iterator.output(2)};
}
```

**Step 3: Run full chebyshev test suite**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__chebyshev_polynomial_t.py -v 2>&1`

Expected: All chebyshev tests pass

**Step 4: Commit**

```bash
git add src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h
git commit -m "refactor(chebyshev): extract chebyshev_polynomial_t_backward_backward_kernel"
```

---

## Task 7: Final verification and cleanup

**Files:**
- Verify: `src/torchscience/csrc/cpu/special_functions/gamma.h`
- Verify: `src/torchscience/csrc/cpu/special_functions/chebyshev_polynomial_t.h`

**Step 1: Run full test suite for both operators**

Run: `cd /Users/goodmaa3/com/github/0x00b1/torchscience/.worktrees/cpu-kernel-refactor && .venv/bin/python -m pytest tests/torchscience/special_functions/test__gamma.py tests/torchscience/special_functions/test__chebyshev_polynomial_t.py -v 2>&1`

Expected: All tests pass

**Step 2: Verify file structure**

Each file should now have exactly:
- 3 kernel functions in anonymous namespace (`*_forward_kernel`, `*_backward_kernel`, `*_backward_backward_kernel`)
- 3 dispatch functions (`*_forward`, `*_backward`, `*_backward_backward`)
- TORCH_LIBRARY_IMPL registration

**Step 3: Mark design doc as complete**

Update `docs/plans/2025-12-27-cpu-special-function-kernel-refactor-design.md` to add `Status: Complete` at the top.

**Step 4: Commit**

```bash
git add docs/plans/2025-12-27-cpu-special-function-kernel-refactor-design.md
git commit -m "docs: mark CPU kernel refactor design as complete"
```
