# CPU Special Function Kernel Refactor Design

**Status: Complete**

## Summary

Refactor CPU special function operators to separate pure scalar kernels from TensorIterator dispatch. Each operator will have exactly three kernel functions (`*_forward_kernel`, `*_backward_kernel`, `*_backward_backward_kernel`) with all math inlined.

## Kernel Structure

Each CPU special function operator has three scalar kernel functions in an anonymous namespace:

```cpp
namespace torchscience::cpu {
namespace {

template <typename T>
T gamma_forward_kernel(T z) {
  // All gamma computation inlined (Lanczos, reflection formula)
}

template <typename T>
T gamma_backward_kernel(T g, T z) {
  // gamma(z) * digamma(z) inlined - no separate helper
}

template <typename T>
std::tuple<T, T> gamma_backward_backward_kernel(T gg, T g, T z) {
  // gamma, digamma, trigamma all inlined
  // Returns (gg_output, new_grad)
}

} // anonymous namespace
} // namespace torchscience::cpu
```

### Naming Convention

- Suffix-based: `*_forward_kernel`, `*_backward_kernel`, `*_backward_backward_kernel`
- No separate helper functions (e.g., no `digamma_kernel`) - all math inlined

### Return Types

- Forward: `T` (single scalar)
- Backward: `T` for single output, `std::tuple<T, T>` for multiple outputs
- Backward_backward: `std::tuple<T, T>` or `std::tuple<T, T, T>` depending on operator arity

## TensorIterator Dispatch Functions

Dispatch functions become thin wrappers that:
1. Set up TensorIterator
2. Dispatch to kernel (no wrapping lambda)

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
    at::kBFloat16, at::kHalf,
    iterator.common_dtype(),
    "gamma_cpu",
    [&] {
      at::native::cpu_kernel(iterator, gamma_forward_kernel<scalar_t>);
    }
  );

  return iterator.output();
}
```

For multiple outputs:

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(
  at::kBFloat16, at::kHalf,
  iterator.common_dtype(),
  "gamma_backward_backward_cpu",
  [&] {
    at::native::cpu_kernel_multiple_outputs(
      iterator,
      gamma_backward_backward_kernel<scalar_t>
    );
  }
);
```

## Scope

Applies to all 4 CPU special function operators:

| Operator | Forward Kernel | Backward Kernel | Backward_Backward Kernel |
|----------|----------------|-----------------|--------------------------|
| gamma | `gamma_forward_kernel(z)` | `gamma_backward_kernel(g, z)` | `gamma_backward_backward_kernel(gg, g, z)` |
| chebyshev_polynomial_t | `chebyshev_polynomial_t_forward_kernel(x, n)` | `chebyshev_polynomial_t_backward_kernel(g, x, n)` | `chebyshev_polynomial_t_backward_backward_kernel(gg_x, gg_n, g, x, n)` |
| hypergeometric_2_f_1 | `hypergeometric_2_f_1_forward_kernel(a, b, c, z)` | `hypergeometric_2_f_1_backward_kernel(g, a, b, c, z)` | `hypergeometric_2_f_1_backward_backward_kernel(...)` |
| incomplete_beta | `incomplete_beta_forward_kernel(a, b, x)` | `incomplete_beta_backward_kernel(g, a, b, x)` | `incomplete_beta_backward_backward_kernel(...)` |

## Out of Scope

- File structure unchanged (each operator in its own `.h` file)
- TORCH_LIBRARY_IMPL registration unchanged
- Meta/Autograd/Autocast backends unaffected
