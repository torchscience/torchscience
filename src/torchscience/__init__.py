from pathlib import Path

import torch

from . import _C

__version__ = "0.0.1"


class _Ops:
    """Wrapper for torchscience operators providing a cleaner API."""

    def __getattr__(self, name: str):
        """Dynamically access operators from torch.ops.torchscience."""
        return getattr(torch.ops.torchscience, name)

    def __dir__(self):
        """List available operators."""
        return dir(torch.ops.torchscience)


# Create ops namespace
ops = _Ops()


def cuda_version():
    """Get the CUDA version used to compile the extension.

    Returns None if CUDA support was not compiled in.
    """
    try:
        return torch.ops.torchscience._cuda_version()
    except (NotImplementedError, RuntimeError):
        # CUDA support not available (CPU-only build)
        return None


def main() -> None:
    print("Hello from torch-science!")
    cuda_ver = cuda_version()
    if cuda_ver is not None:
        print(f"CUDA version: {cuda_ver}")
    else:
        print("CUDA support: not available (CPU-only build)")
