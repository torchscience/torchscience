from pathlib import Path

import torch

from . import _C


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
    """Get the CUDA version used to compile the extension."""
    return torch.ops.torchscience._cuda_version()


def main() -> None:
    print("Hello from torch-science!")
    print(f"CUDA version: {cuda_version()}")
