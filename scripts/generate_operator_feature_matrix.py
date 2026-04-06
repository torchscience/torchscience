"""Generate operator feature matrix from runtime introspection."""

from __future__ import annotations

from pathlib import Path

import torch


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

    output_path.write_text("\n".join(sections))
    print(f"Wrote {output_path} ({len(ops)} operators)")


if __name__ == "__main__":
    main()
