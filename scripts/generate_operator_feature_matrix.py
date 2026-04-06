"""Generate operator feature matrix from runtime introspection."""

from __future__ import annotations

from pathlib import Path

import torch

import torchscience.special_functions  # noqa: F401  – triggers dispatcher registration

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


def get_op_schema_args(op: str) -> list[tuple[str, str]]:
    """Return [(name, type_str), ...] for all schema arguments of an operator."""
    op_func = getattr(torch.ops.torchscience, op)
    schema = op_func.default._schema
    return [(a.name, str(a.type)) for a in schema.arguments]


def build_test_inputs(args: list[tuple[str, str]], dtype: torch.dtype) -> list:
    """Build test arguments matching the schema.

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
    """For each op, probe which dtypes are supported by calling with test tensors.

    Returns {op_name: {dtype_name: bool}}.
    """
    results: dict[str, dict[str, bool]] = {}
    for op in ops:
        args = get_op_schema_args(op)
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
    separator = (
        "|----------|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
    )
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

    output_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "operator-feature-matrix.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append("# Operator Feature Matrix\n")
    sections.append(
        f"Auto-generated from runtime introspection of {len(ops)} operators.\n"
    )
    sections.append("Cells marked `x` indicate the feature is implemented.\n")

    sections.append(
        format_table(
            ops,
            ["CPU", "CUDA", "Meta", "Autograd", "Autocast"],
            backends,
            "Backends",
        )
    )

    dtype_names = [name for name, _ in DTYPES]
    dtypes = detect_dtypes(ops)

    sections.append(format_table(ops, dtype_names, dtypes, "Dtypes"))

    output_path.write_text("\n".join(sections))
    print(f"Wrote {output_path} ({len(ops)} operators)")


if __name__ == "__main__":
    main()
