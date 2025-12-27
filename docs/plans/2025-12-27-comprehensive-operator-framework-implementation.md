# Comprehensive Operator Framework - Implementation Plan

## Executive Summary

Migrate all manual operator schemas in `torchscience.cpp` to X-macro-based infrastructure, extending the pattern established for `special_functions.def` to cover all operator categories per ezyang's taxonomy.

**Current State:** 22 manual `module.def()` calls in `torchscience.cpp`
**Target State:** 0 manual `module.def()` calls; all operators registered via X-macros

---

## Operators to Migrate

| Operator | Category | Wave |
|----------|----------|------|
| `gamma`, `chebyshev_polynomial_t`, `incomplete_beta`, `hypergeometric_2_f_1` | TensorIterator/Pointwise | Already done |
| `kurtosis` | TensorIterator/Reduction | Wave 2 |
| `hilbert_transform`, `inverse_hilbert_transform` | Fixed/Transform | Wave 3 |
| `butterworth_analog_bandpass_filter` | Fixed/Transform | Wave 3 |
| `minkowski_distance` | N-Dimensional/Pairwise | Wave 4 |
| `cook_torrance` | Batched/Graphics | Wave 5 |
| `sine_wave`, `rectangular_window` | Factory | Wave 6 |
| `histogram`, `histogram_edges` | Dynamic | Wave 7 |
| `rosenbrock` | Composite | Wave 8 (optional - already uses CompositeImplicitAutograd) |

---

## Wave 1: Foundation (Directory Reorganization)

**Goal:** Create subdirectory structure in `core/` and move pointwise infrastructure.

### Task 1.1: Create tensoriterator subdirectory
- [ ] Create `src/torchscience/csrc/core/tensoriterator/` directory
- **Files:** None yet (directory only)
- **Verification:** Directory exists

### Task 1.2: Move pointwise_registration.h
- [ ] Move `core/pointwise_registration.h` to `core/tensoriterator/pointwise_registration.h`
- [ ] Update include in `cpu/special_functions.h`: `#include "../core/tensoriterator/pointwise_registration.h"`
- [ ] Update include in `meta/special_functions.h`
- [ ] Update include in `autograd/special_functions.h`
- [ ] Update include in `autocast/special_functions.h`
- **Verification:** `uv run pytest tests/torchscience/special_functions/ -x`

### Task 1.3: Move schema_generation.h
- [ ] Move `core/schema_generation.h` to `core/tensoriterator/pointwise_schema.h`
- [ ] Update include in `torchscience.cpp`: `#include "core/tensoriterator/pointwise_schema.h"`
- **Verification:** Build succeeds, tests pass

### Task 1.4: Rename creation_common.h to common.h
- [ ] Rename `core/creation_common.h` to `core/common.h`
- [ ] Update all includes (9 files reference it)
- **Verification:** Build succeeds

### Task 1.5: Create remaining subdirectories
- [ ] Create `core/reduction/`
- [ ] Create `core/fixed/`
- [ ] Create `core/batched/`
- [ ] Create `core/ndimensional/`
- [ ] Create `core/factory/`
- [ ] Create `core/dynamic/`
- **Verification:** Directories exist

**Wave 1 Completion Criteria:**
- All pointwise infrastructure in `core/tensoriterator/`
- `dispatch_helpers.h` remains at `core/` level (shared)
- `common.h` at `core/` level (shared)
- All existing tests pass
- No behavior changes

---

## Wave 2: Reduction X-Macro Infrastructure

**Goal:** Create X-macro system for reduction operators, migrate `kurtosis`.

### Task 2.1: Create reduction_schema.h
- [ ] Create `core/reduction/reduction_schema.h`
- [ ] Implement `ReductionSchema` template with specializations for 0, 1, 2 extra args
- [ ] Include schema generators for forward, backward, backward_backward
- **File content pattern:**
```cpp
template<std::size_t NumExtraArgs>
struct ReductionSchema;

template<>
struct ReductionSchema<2> {
    static std::string forward(const char* name, const char* extra_args) {
        // "kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor"
    }
};
```

### Task 2.2: Create reduction_registration.h
- [ ] Create `core/reduction/reduction_registration.h`
- [ ] Move `CPUReductionOperator` from `cpu/reduction_operators.h` to new location
- [ ] Create `MetaReductionOperator` (shape inference only)
- [ ] Create `AutogradReductionOperator` (gradient wrapper)
- [ ] Create `AutocastReductionOperator` (mixed precision)
- [ ] Create arity-based registrar templates

### Task 2.3: Create reductions.def
- [ ] Create `src/torchscience/csrc/operators/reductions.def`
- [ ] Define `TORCHSCIENCE_REDUCTIONS(X)` macro
- [ ] Add kurtosis entry: `X(kurtosis, KurtosisImpl, "bool fisher, bool bias")`

### Task 2.4: Create kurtosis traits struct
- [ ] Create `impl/statistics/kurtosis_traits.h`
- [ ] Define `KurtosisImpl` with `dispatch_forward`, `dispatch_backward`, `dispatch_backward_backward`
- [ ] Use `DECLARE_OP_NAME(kurtosis)` pattern from special_functions

### Task 2.5: Update backend registration files
- [ ] Create `cpu/statistics/kurtosis.h` using X-macro expansion
- [ ] Create `meta/statistics/kurtosis.h` using X-macro expansion
- [ ] Create `autograd/statistics/kurtosis.h` using X-macro expansion
- [ ] Create `autocast/statistics/kurtosis.h` using X-macro expansion

### Task 2.6: Update torchscience.cpp
- [ ] Add `#include "operators/reductions.def"`
- [ ] Replace manual kurtosis schema with:
```cpp
#define DEFINE_REDUCTION(name, impl, extra_args) \
    DEFINE_REDUCTION_SCHEMA(module, name, extra_args);
TORCHSCIENCE_REDUCTIONS(DEFINE_REDUCTION)
#undef DEFINE_REDUCTION
```
- [ ] Remove 3 manual `module.def()` lines for kurtosis

### Task 2.7: Verification
- [ ] `uv run pytest tests/torchscience/statistics/descriptive/test__kurtosis.py -v`
- [ ] Verify gradcheck passes
- [ ] Verify backward_backward (second-order gradients) work

**Wave 2 Completion Criteria:**
- `reductions.def` exists with kurtosis
- kurtosis registered via X-macro
- 3 fewer manual `module.def()` in torchscience.cpp
- All kurtosis tests pass

---

## Wave 3: Transform X-Macro Infrastructure

**Goal:** Create X-macro for fixed-dimension transform operators.

### Task 3.1: Create fixed_schema.h
- [ ] Create `core/fixed/fixed_schema.h`
- [ ] Implement `TransformSchema` for operators with `(Tensor, int n, int dim, ...extra_params)`

### Task 3.2: Create fixed_registration.h
- [ ] Move `CPUFixedOperator` from `cpu/fixed_operators.h`
- [ ] Create Meta, Autograd, Autocast variants
- [ ] Create registrar templates

### Task 3.3: Create transforms.def
- [ ] Define `TORCHSCIENCE_TRANSFORMS(X)` macro
- [ ] Add entries for:
  - `hilbert_transform`
  - `inverse_hilbert_transform`
  - `butterworth_analog_bandpass_filter`

### Task 3.4: Create transform traits structs
- [ ] Create `impl/integral_transform/hilbert_transform_traits.h`
- [ ] Create `impl/integral_transform/inverse_hilbert_transform_traits.h`
- [ ] Create `impl/signal_processing/butterworth_traits.h`

### Task 3.5: Update backend files and torchscience.cpp
- [ ] Create CPU/Meta/Autograd/Autocast registration files
- [ ] Replace 9 manual `module.def()` lines in torchscience.cpp

### Task 3.6: Verification
- [ ] All integral_transform tests pass
- [ ] All signal_processing filter tests pass

**Wave 3 Completion Criteria:**
- `transforms.def` exists with 3 operators
- 9 fewer manual `module.def()` in torchscience.cpp

---

## Wave 4: Distance/Pairwise X-Macro Infrastructure

**Goal:** Create X-macro for pairwise distance operators.

### Task 4.1: Create ndimensional/pairwise_schema.h
- [ ] Schema for `(Tensor x, Tensor y, ...extra_params) -> Tensor`

### Task 4.2: Create pairwise_registration.h
- [ ] Move `CPUPairwiseOperator` from `cpu/pairwise_operators.h`
- [ ] Create backend variants

### Task 4.3: Create distance.def
- [ ] Add `minkowski_distance` entry

### Task 4.4: Create minkowski_distance_traits.h
- [ ] Define `MinkowskiDistanceImpl`

### Task 4.5: Update registration and torchscience.cpp
- [ ] Replace 2 manual `module.def()` lines

### Task 4.6: Verification
- [ ] Distance tests pass

**Wave 4 Completion Criteria:**
- `distance.def` exists
- 2 fewer manual `module.def()`

---

## Wave 5: Graphics X-Macro Infrastructure

**Goal:** Handle 5-ary batched operators like cook_torrance.

### Task 5.1: Extend arity system
- [ ] Add `CPUQuinaryOperator` (5 inputs) to operators.h pattern
- [ ] Or: Use batched operator pattern with fixed trailing dims

### Task 5.2: Create graphics.def
- [ ] Add `cook_torrance` entry

### Task 5.3: Create cook_torrance_traits.h
- [ ] Define `CookTorranceImpl`

### Task 5.4: Update registration and torchscience.cpp
- [ ] Replace 3 manual `module.def()` lines

### Task 5.5: Verification
- [ ] Graphics shading tests pass

**Wave 5 Completion Criteria:**
- `graphics.def` exists
- 3 fewer manual `module.def()`

---

## Wave 6: Factory X-Macro Infrastructure

**Goal:** Create X-macro for tensor creation operators.

### Task 6.1: Create factory/factory_schema.h
- [ ] Schema with TensorOptions pattern (dtype, layout, device, requires_grad)

### Task 6.2: Create factory_registration.h
- [ ] Only CPU and Meta needed (factories don't need Autograd/Autocast)

### Task 6.3: Create creation.def
- [ ] Add `sine_wave`, `rectangular_window`

### Task 6.4: Update torchscience.cpp
- [ ] Replace 2 manual `module.def()` lines

### Task 6.5: Verification
- [ ] Signal processing waveform tests pass
- [ ] Window function tests pass

**Wave 6 Completion Criteria:**
- `creation.def` exists
- 2 fewer manual `module.def()`

---

## Wave 7: Dynamic Operators

**Goal:** Handle data-dependent output shape operators.

### Task 7.1: Create dynamic/dynamic_schema.h
- [ ] Flexible schema for multi-output operators

### Task 7.2: Create histogram entries
- [ ] Note: histogram is non-differentiable, simpler registration
- [ ] May not need full X-macro (benefit is marginal for non-differentiable ops)

### Task 7.3: Evaluate effort vs benefit
- [ ] If 2 operators don't justify full infrastructure, leave as manual
- [ ] Document decision

**Wave 7 Completion Criteria:**
- Decision documented
- histogram handled appropriately

---

## Wave 8: Documentation

**Goal:** Comprehensive documentation for all operator categories.

### Task 8.1: Update adding-operators.md
- [ ] Add section for each operator category
- [ ] Include category-specific examples
- [ ] Document traits struct requirements per category

### Task 8.2: Create category guides
- [ ] `docs/contributing/adding-reduction-operators.md`
- [ ] `docs/contributing/adding-transform-operators.md`
- [ ] `docs/contributing/adding-distance-operators.md`

### Task 8.3: Update architecture docs
- [ ] Document core/ directory structure
- [ ] Document X-macro patterns

**Wave 8 Completion Criteria:**
- Documentation covers all categories
- New contributor can add any operator type

---

## Final Verification

- [ ] `uv run pytest tests/torchscience/ -v` - All tests pass
- [ ] `torchscience.cpp` has 0 manual `module.def()` (or documented exceptions)
- [ ] Each operator category has example in documentation
- [ ] Build times not significantly regressed

---

## Risk Mitigation

1. **Include path complexity:** Use relative paths consistently, test across platforms
2. **Template instantiation times:** Monitor build times, consider explicit instantiations if needed
3. **Backward compatibility:** Keep Python API identical, only internal reorganization
4. **Incomplete migrations:** Can leave specific operators as manual if X-macro doesn't fit

---

## Batch Execution Suggestions

**Batch A (Foundation):** Tasks 1.1-1.5 - Pure refactoring, no logic changes
**Batch B (Reduction):** Tasks 2.1-2.7 - Self-contained, can verify independently
**Batch C (Transform):** Tasks 3.1-3.6 - Builds on reduction pattern
**Batch D (Distance + Graphics):** Tasks 4.1-5.5 - Similar patterns
**Batch E (Factory + Dynamic + Docs):** Tasks 6.1-8.3 - Final cleanup
