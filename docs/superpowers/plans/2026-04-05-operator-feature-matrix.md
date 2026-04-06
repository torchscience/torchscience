# Operator Feature Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Python script that introspects the built torchscience library at runtime and generates a Markdown document with five tables showing which PyTorch features each operator supports.

**Architecture:** Single Python script (`scripts/generate_operator_feature_matrix.py`) with no external dependencies beyond torch and torchscience. The script discovers operators via PyTorch's dispatcher API, probes dtype support by calling operators with test tensors, and writes structured Markdown to `docs/operator-feature-matrix.md`.

**Tech Stack:** Python 3, torch (dispatcher internals), torchscience (built library)

**Spec:** `docs/superpowers/specs/2026-04-05-operator-feature-matrix-design.md`

---

### Task 1: Operator discovery and backend detection

**Files:**
- Create: `scripts/generate_operator_feature_matrix.py`

This task builds the core of the script: discovering all forward operators and detecting which dispatch keys each one supports. The script will output just Table 1 (Backends) initially.

- [ ] **Step 1: Create the script with operator discovery and backend detection**

Create `scripts/generate_operator_feature_matrix.py`:

```python
"""Generate operator feature matrix from runtime introspection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchscience.special_functions  # triggers C++ library load


def discover_forward_ops() -> list[str]:
    """Return sorted list of forward operator names in the torchscience namespace."""
    all_regs = torch._C._dispatch_get_registrations_for_dispatch_key("CPU")
    ts_ops = [r for r in all_regs if r.startswith("torchscience::")]
    return sorted(
        set(
            op.replace("torchscience::", "")
            for op in ts_ops
            if "_backward" not in op
        )
    )


def has_dispatch_key(op: str, key: str) -> bool:
    """Check if an operator has a kernel registered for the given dispatch key."""
    return torch._C._dispatch_has_kernel_for_dispatch_key(
        f"torchscience::{op}", key
    )


def detect_backends(ops: list[str]) -> dict[str, dict[str, bool]]:
    """For each op, detect which backend dispatch keys are registered.

    Returns {op_name: {column_name: bool}}.
    """
    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        results[op] = {
            "CPU": has_dispatch_key(op, "CPU"),
            "CUDA": has_dispatch_key(op, "CUDA"),
            "Meta": has_dispatch_key(op, "Meta"),
            "Autograd": has_dispatch_key(op, "Autograd"),
            "Autocast": (
                has_dispatch_key(op, "AutocastCPU")
                or has_dispatch_key(op, "AutocastCUDA")
            ),
        }
    return results


def format_table(
    ops: list[str],
    columns: list[str],
    data: dict[str, dict[str, bool]],
    title: str,
) -> str:
    """Format a feature matrix table as Markdown."""
    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")

    # Header
    header = "| Operator | " + " | ".join(columns) + " |"
    separator = "|----------|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for op in ops:
        cells = []
        for col in columns:
            cells.append(" x " if data[op].get(col, False) else "   ")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")

    # Summary row
    total = len(ops)
    counts = []
    for col in columns:
        n = sum(1 for op in ops if data[op].get(col, False))
        pct = round(100 * n / total) if total else 0
        counts.append(f" {n}/{total} ({pct}%) ")
    lines.append(f"| **Total** | " + " | ".join(counts) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ops = discover_forward_ops()

    backends = detect_backends(ops)

    output_path = Path(__file__).resolve().parent.parent / "docs" / "operator-feature-matrix.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append("# Operator Feature Matrix\n")
    sections.append(
        f"Auto-generated from runtime introspection of {len(ops)} operators.\n"
    )
    sections.append(
        "Cells marked `x` indicate the feature is implemented.\n"
    )

    sections.append(
        format_table(ops, ["CPU", "CUDA", "Meta", "Autograd", "Autocast"], backends, "Backends")
    )

    output_path.write_text("\n".join(sections))
    print(f"Wrote {output_path} ({len(ops)} operators)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify Table 1 output**

Run:
```bash
uv run python scripts/generate_operator_feature_matrix.py
```

Expected: Script prints `Wrote .../docs/operator-feature-matrix.md (140 operators)`. The file contains a Backends table with 140 rows. CPU, Meta, Autograd, Autocast columns should be mostly `x`. CUDA may be empty if built without CUDA.

Verify:
```bash
head -20 docs/operator-feature-matrix.md
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_operator_feature_matrix.py docs/operator-feature-matrix.md
git commit -m "feat: add operator feature matrix script with backend detection"
```

---

### Task 2: Dtype detection

**Files:**
- Modify: `scripts/generate_operator_feature_matrix.py`

Add dtype probing by calling each operator with test tensors of each dtype and catching errors.

- [ ] **Step 1: Add dtype detection to the script**

Add the following functions to `scripts/generate_operator_feature_matrix.py`, above `main()`:

```python
DTYPES = [
    ("bool", torch.bool),
    ("int8", torch.int8),
    ("int16", torch.int16),
    ("int32", torch.int32),
    ("int64", torch.int64),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
    ("float64", torch.float64),
    ("complex32", torch.complex32),
    ("complex64", torch.complex64),
    ("complex128", torch.complex128),
]


def get_op_tensor_arity(op: str) -> tuple[int, list[tuple[str, str]]]:
    """Return (tensor_arg_count, [(name, type_str), ...]) for all schema args.

    Non-tensor args are included so we can provide default values.
    """
    op_func = getattr(torch.ops.torchscience, op)
    schema = op_func.default._schema
    args = []
    for a in schema.arguments:
        args.append((a.name, str(a.type)))
    return args


def build_test_inputs(
    args: list[tuple[str, str]], dtype: torch.dtype
) -> list:
    """Build a list of test arguments matching the schema.

    Tensor args get a ones(1) tensor of the target dtype.
    int args get 1. float args get 1.0.
    """
    inputs = []
    for name, type_str in args:
        if type_str == "Tensor":
            inputs.append(torch.ones(1, dtype=dtype))
        elif type_str == "int":
            inputs.append(1)
        elif type_str == "float":
            inputs.append(1.0)
        else:
            inputs.append(torch.ones(1, dtype=dtype))
    return inputs


def detect_dtypes(ops: list[str]) -> dict[str, dict[str, bool]]:
    """For each op, probe which dtypes are supported.

    Returns {op_name: {dtype_name: bool}}.
    """
    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        args = get_op_tensor_arity(op)
        op_func = getattr(torch.ops.torchscience, op)
        results[op] = {}
        for dtype_name, dtype in DTYPES:
            inputs = build_test_inputs(args, dtype)
            try:
                op_func(*inputs)
                results[op][dtype_name] = True
            except (RuntimeError, NotImplementedError):
                results[op][dtype_name] = False
    return results
```

Then in `main()`, after the backends section, add:

```python
    dtype_names = [name for name, _ in DTYPES]
    dtypes = detect_dtypes(ops)

    sections.append(
        format_table(ops, dtype_names, dtypes, "Dtypes")
    )
```

- [ ] **Step 2: Run the script and verify Table 2 output**

Run:
```bash
uv run python scripts/generate_operator_feature_matrix.py
```

Expected: The output now has two tables. The Dtypes table should show `x` for float16, bfloat16, float32, float64 for most operators. complex64 and complex128 should be `x` only for operators that used the `_WITH_COMPLEX` macro. bool, int*, and complex32 should be mostly empty.

Verify:
```bash
grep -A 5 "## Dtypes" docs/operator-feature-matrix.md | head -10
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_operator_feature_matrix.py docs/operator-feature-matrix.md
git commit -m "feat: add dtype detection to operator feature matrix"
```

---

### Task 3: Hessian, Masked, Nested, Named detection (Other table)

**Files:**
- Modify: `scripts/generate_operator_feature_matrix.py`

- [ ] **Step 1: Add Other table detection**

Add the following function to `scripts/generate_operator_feature_matrix.py`, above `main()`:

```python
def detect_other(ops: list[str]) -> dict[str, dict[str, bool]]:
    """Detect Hessian, Masked, Nested, Named support.

    Returns {op_name: {column_name: bool}}.
    """
    # Build set of all CPU registrations for backward_backward check
    all_cpu = set(
        r.replace("torchscience::", "")
        for r in torch._C._dispatch_get_registrations_for_dispatch_key("CPU")
        if r.startswith("torchscience::")
    )

    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        results[op] = {
            "Hessian": f"{op}_backward_backward" in all_cpu,
            "Masked": False,  # no dispatch key available
            "Nested": (
                has_dispatch_key(op, "NestedTensorCPU")
                or has_dispatch_key(op, "NestedTensorCUDA")
            ),
            "Named": False,  # no dispatch key available
        }
    return results
```

Then in `main()`, after the dtypes section, add:

```python
    other = detect_other(ops)

    sections.append(
        format_table(ops, ["Hessian", "Masked", "Nested", "Named"], other, "Other")
    )
```

- [ ] **Step 2: Run the script and verify Table 4 output**

Run:
```bash
uv run python scripts/generate_operator_feature_matrix.py
```

Expected: The Other table should show `x` for Hessian on all 140 operators (all have backward_backward kernels). Masked, Nested, Named should be empty.

Verify:
```bash
grep -A 5 "## Other" docs/operator-feature-matrix.md | head -10
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_operator_feature_matrix.py docs/operator-feature-matrix.md
git commit -m "feat: add hessian/masked/nested/named detection to feature matrix"
```

---

### Task 4: Sparse and Quantized tables

**Files:**
- Modify: `scripts/generate_operator_feature_matrix.py`

- [ ] **Step 1: Add Sparse and Quantized detection**

Add the following functions to `scripts/generate_operator_feature_matrix.py`, above `main()`:

```python
def detect_sparse(ops: list[str]) -> dict[str, dict[str, bool]]:
    """Detect sparse backend support.

    Returns {op_name: {column_name: bool}}.
    """
    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        results[op] = {
            "Sparse COO CPU": has_dispatch_key(op, "SparseCPU"),
            "Sparse COO CUDA": has_dispatch_key(op, "SparseCUDA"),
            "Sparse CSR CPU": has_dispatch_key(op, "SparseCsrCPU"),
            "Sparse CSR CUDA": has_dispatch_key(op, "SparseCsrCUDA"),
        }
    return results


def detect_quantized(ops: list[str]) -> dict[str, dict[str, bool]]:
    """Detect quantized backend support.

    Returns {op_name: {column_name: bool}}.
    """
    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        results[op] = {
            "Quantized CPU": has_dispatch_key(op, "QuantizedCPU"),
            "Quantized CUDA": has_dispatch_key(op, "QuantizedCUDA"),
        }
    return results
```

Then in `main()`, insert the sparse table after the Other table and add the quantized table last:

```python
    sparse = detect_sparse(ops)

    sections.append(
        format_table(
            ops,
            ["Sparse COO CPU", "Sparse COO CUDA", "Sparse CSR CPU", "Sparse CSR CUDA"],
            sparse,
            "Sparse",
        )
    )

    quantized = detect_quantized(ops)

    sections.append(
        format_table(
            ops,
            ["Quantized CPU", "Quantized CUDA"],
            quantized,
            "Quantized",
        )
    )
```

The final table order in `main()` should be: Backends, Dtypes, Sparse, Other, Quantized — matching the spec's Table 1–5 ordering.

- [ ] **Step 2: Run the script and verify all five tables**

Run:
```bash
uv run python scripts/generate_operator_feature_matrix.py
```

Expected: Output now has all five tables. Sparse and Quantized tables should show `x` for only a handful of operators (gamma, beta, chebyshev_polynomial_t, incomplete_beta, hypergeometric_2_f_1 were the ones registered in the sparse/quantized backends).

Verify:
```bash
grep "^## " docs/operator-feature-matrix.md
```

Expected output:
```
## Backends
## Dtypes
## Sparse
## Other
## Quantized
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_operator_feature_matrix.py docs/operator-feature-matrix.md
git commit -m "feat: add sparse and quantized tables to operator feature matrix"
```

---

### Task 5: Final validation

**Files:**
- Read: `docs/operator-feature-matrix.md`
- Read: `docs/superpowers/specs/2026-04-05-operator-feature-matrix-design.md`

- [ ] **Step 1: Validate output against spec**

Run the script one final time and check the output:

```bash
uv run python scripts/generate_operator_feature_matrix.py
```

Verify:
1. Five tables exist (Backends, Dtypes, Sparse, Other, Quantized)
2. Each table has 140 operator rows + 1 summary row
3. Summary rows show `N/M (P%)` format
4. Cells use `x` for implemented, spaces for not implemented
5. Operators are sorted alphabetically
6. The Autocast column is marked `x` when either AutocastCPU or AutocastCUDA has a kernel

```bash
# Count tables
grep -c "^## " docs/operator-feature-matrix.md
# Expected: 5

# Count data rows in first table (excluding header, separator, summary)
grep -c "^| [a-z]" docs/operator-feature-matrix.md | head -1
# Should see 140 per table = 700 total

# Check summary row exists
grep "Total" docs/operator-feature-matrix.md
```

- [ ] **Step 2: Commit the final output**

```bash
git add docs/operator-feature-matrix.md
git commit -m "docs: regenerate operator feature matrix"
```
