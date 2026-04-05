# Operator Feature Matrix

Auto-generated Markdown table tracking which PyTorch features each operator supports.

## Purpose

Provide a single-glance view of implementation coverage across all operators and all PyTorch integration features. The table is generated from source by a Python script so it stays in sync with the codebase.

## Output

A Markdown file at `docs/operator-feature-matrix.md` containing:

1. A table where rows are operators and columns are features
2. A summary row at the bottom with counts and percentages per column

### Rows

All forward operators extracted from schema definitions in `src/torchscience/csrc/special_functions.cpp`. An operator is any `m.def(...)` call whose name does not contain `_backward`. Rows are sorted alphabetically by operator name.

### Columns (27)

#### Backends (5)

| Column | Detection |
|--------|-----------|
| CPU | `TORCHSCIENCE_CPU_POINTWISE_*_OPERATOR` macro in `csrc/cpu/special_functions.h` |
| CUDA | `TORCHSCIENCE_CUDA_POINTWISE_*_OPERATOR` macro in `csrc/cuda/special_functions.cu` |
| Meta | `TORCHSCIENCE_META_POINTWISE_*_OPERATOR` macro in `csrc/meta/special_functions.h` |
| Autograd | `TORCHSCIENCE_AUTOGRAD_POINTWISE_*_OPERATOR` macro in `csrc/autograd/special_functions.h` |
| Autocast | `TORCHSCIENCE_AUTOCAST_POINTWISE_*_OPERATOR` macro in `csrc/autocast/special_functions.h` |

#### Dtypes (12)

| Column | Detection |
|--------|-----------|
| bool | `AT_DISPATCH_ALL_TYPES` or bool-specific dispatch. Not dispatched by current macros. |
| int8 | Integer dispatch. Not dispatched by current macros. |
| int16 | Integer dispatch. Not dispatched by current macros. |
| int32 | Integer dispatch. Not dispatched by current macros. |
| int64 | Integer dispatch. Not dispatched by current macros. |
| float16 | `AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, ...)`. All operators with a CPU registration. |
| bfloat16 | `AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, ...)`. All operators with a CPU registration. |
| float32 | `AT_DISPATCH_FLOATING_TYPES`. All operators with a CPU registration. |
| float64 | `AT_DISPATCH_FLOATING_TYPES`. All operators with a CPU registration. |
| complex32 | `c10::complex<Half>`. Not dispatched by current macros. |
| complex64 | `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES`. Operator uses `_WITH_COMPLEX` macro suffix. |
| complex128 | `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES`. Operator uses `_WITH_COMPLEX` macro suffix. |

#### Sparse (4)

| Column | Detection |
|--------|-----------|
| Sparse COO CPU | `REGISTER_SPARSE_COO_CPU_*` macro in `csrc/sparse/coo/cpu/special_functions.h` |
| Sparse COO CUDA | `REGISTER_SPARSE_COO_CUDA_*` macro in `csrc/sparse/coo/cuda/special_functions.h` |
| Sparse CSR CPU | `REGISTER_SPARSE_CSR_CPU_*` macro in `csrc/sparse/csr/cpu/special_functions.h` |
| Sparse CSR CUDA | `REGISTER_SPARSE_CSR_CUDA_*` macro in `csrc/sparse/csr/cuda/special_functions.h` |

#### Other (4)

| Column | Detection |
|--------|-----------|
| Hessian | Kernel file `<op>_backward_backward.h` exists in `csrc/kernel/special_functions/` |
| Masked | Masked dispatch key registration. Not implemented today. |
| Nested | NestedTensor dispatch key registration. Not implemented today. |
| Named | Named dispatch key registration. Not implemented today. |

#### Quantized (2)

| Column | Detection |
|--------|-----------|
| Quantized CPU | `REGISTER_QUANTIZED_CPU_*` macro in `csrc/quantized/cpu/special_functions.h` |
| Quantized CUDA | `REGISTER_QUANTIZED_CUDA_*` macro in `csrc/quantized/cuda/special_functions.h` |

## Script

A Python script at `scripts/generate_operator_feature_matrix.py` that:

1. Parses the schema file to extract operator names
2. Scans each backend registration file for macro invocations matching each operator
3. Checks kernel directory for `_backward_backward.h` files
4. Detects `_WITH_COMPLEX` suffix on CPU macros for complex dtype support
5. Infers float16/bfloat16/float32/float64 support from CPU registration presence
6. Writes the Markdown table to `docs/operator-feature-matrix.md`

### Detection Patterns

```python
# Schema: extract operator name from m.def("op_name(...)")
SCHEMA_RE = r'm\.def\("(\w+)\('

# Backend macros: extract operator name as first argument
BACKEND_RE = {
    "cpu":      r'TORCHSCIENCE_CPU_POINTWISE_\w+_OPERATOR(?:_WITH_COMPLEX)?\((\w+),',
    "cuda":     r'TORCHSCIENCE_CUDA_POINTWISE_\w+_OPERATOR(?:_WITH_COMPLEX)?\((\w+),',
    "meta":     r'TORCHSCIENCE_META_POINTWISE_\w+_OPERATOR\((\w+),',
    "autograd": r'TORCHSCIENCE_AUTOGRAD_POINTWISE_\w+_OPERATOR\((\w+),',
    "autocast": r'TORCHSCIENCE_AUTOCAST_POINTWISE_\w+_OPERATOR\((\w+),',
}

# Complex: operator uses _WITH_COMPLEX suffix in CPU file
COMPLEX_RE = r'TORCHSCIENCE_CPU_POINTWISE_\w+_OPERATOR_WITH_COMPLEX\((\w+),'

# Sparse/Quantized: operator name from registration macro
SPARSE_COO_CPU_RE  = r'REGISTER_SPARSE_COO_CPU_\w+\(m,\s*(\w+)\)'
SPARSE_COO_CUDA_RE = r'REGISTER_SPARSE_COO_CUDA_\w+\(m,\s*(\w+)\)'
SPARSE_CSR_CPU_RE  = r'REGISTER_SPARSE_CSR_CPU_\w+\(m,\s*(\w+)\)'
SPARSE_CSR_CUDA_RE = r'REGISTER_SPARSE_CSR_CUDA_\w+\(m,\s*(\w+)\)'
QUANTIZED_CPU_RE   = r'REGISTER_QUANTIZED_CPU_\w+\(m,\s*(\w+)\)'
QUANTIZED_CUDA_RE  = r'REGISTER_QUANTIZED_CUDA_\w+\(m,\s*(\w+)\)'

# Hessian: check file existence
# Path: csrc/kernel/special_functions/{op}_backward_backward.h
```

### Source Files

| Detection target | File path |
|-----------------|-----------|
| Schema | `src/torchscience/csrc/special_functions.cpp` |
| CPU | `src/torchscience/csrc/cpu/special_functions.h` |
| CUDA | `src/torchscience/csrc/cuda/special_functions.cu` |
| Meta | `src/torchscience/csrc/meta/special_functions.h` |
| Autograd | `src/torchscience/csrc/autograd/special_functions.h` |
| Autocast | `src/torchscience/csrc/autocast/special_functions.h` |
| Sparse COO CPU | `src/torchscience/csrc/sparse/coo/cpu/special_functions.h` |
| Sparse COO CUDA | `src/torchscience/csrc/sparse/coo/cuda/special_functions.h` |
| Sparse CSR CPU | `src/torchscience/csrc/sparse/csr/cpu/special_functions.h` |
| Sparse CSR CUDA | `src/torchscience/csrc/sparse/csr/cuda/special_functions.h` |
| Quantized CPU | `src/torchscience/csrc/quantized/cpu/special_functions.h` |
| Quantized CUDA | `src/torchscience/csrc/quantized/cuda/special_functions.h` |
| Hessian kernels | `src/torchscience/csrc/kernel/special_functions/*_backward_backward.h` |

### Table Layout

The output is split into five separate tables, each with its own summary row. Every table shares the same Operator column (rows sorted alphabetically).

#### Table 1: Backends

```markdown
| Operator | CPU | CUDA | Meta | Autograd | Autocast |
|----------|-----|------|------|----------|----------|
| gamma    | x   | x    | x    | x        | x        |
```

#### Table 2: Dtypes

```markdown
| Operator | bool | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 | complex32 | complex64 | complex128 |
|----------|------|------|-------|-------|-------|---------|----------|---------|---------|-----------|-----------|------------|
| gamma    |      |      |       |       |       | x       | x        | x       | x       |           | x         | x          |
```

#### Table 3: Sparse

```markdown
| Operator | Sparse COO CPU | Sparse COO CUDA | Sparse CSR CPU | Sparse CSR CUDA |
|----------|----------------|-----------------|----------------|-----------------|
| gamma    |                |                 |                |                 |
```

#### Table 4: Other

```markdown
| Operator | Hessian | Masked | Nested | Named |
|----------|---------|--------|--------|-------|
| gamma    | x       |        |        |       |
```

#### Table 5: Quantized

```markdown
| Operator | Quantized CPU | Quantized CUDA |
|----------|---------------|----------------|
| gamma    |               |                |
```

Cells use `x` for implemented, empty for not implemented.

### Summary Row

Each table includes a summary row at the bottom showing `N/M (P%)` where N is the count of operators with the feature, M is total operators, and P is the percentage.

## Usage

```bash
uv run python scripts/generate_operator_feature_matrix.py
```

No dependencies beyond the standard library. The script uses only `re`, `pathlib`, and string operations.
