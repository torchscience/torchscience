"""Shared dispatch helper for the colored-noise creation operators.

All five colored-noise ops (white/pink/brown/blue/violet) share the same
calling convention and the same anchor-tensor dispatch pattern. Centralizing
that logic here keeps the per-color modules focused on documentation of the
specific spectral characteristics rather than mechanical PyTorch plumbing.
"""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers torch.ops.torchscience


def dispatch_colored_noise(
    op_name: str,
    op,
    size: int,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format=torch.contiguous_format,
) -> Tensor:
    """Validate kwargs, build the anchor tensor, and dispatch to ``op``.

    Parameters mirror the public API of every colored-noise function. The
    ``op_name`` is used only for clearer error messages; ``op`` is the actual
    ``torch.ops.torchscience.<name>`` callable.
    """
    if out is not None:
        raise NotImplementedError(f"{op_name}: out= is not supported")
    if layout != torch.strided:
        raise ValueError(f"{op_name}: only strided layout is supported")
    if memory_format != torch.contiguous_format:
        raise ValueError(f"{op_name}: only contiguous_format is supported")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = dtype if dtype is not None else torch.get_default_dtype()
    anchor_kw = {}
    if pin_memory and dev.type == "cpu":
        anchor_kw["pin_memory"] = True
    anchor = torch.empty((), device=dev, dtype=dt, **anchor_kw)

    return op(
        anchor,
        size,
        generator,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
    )
