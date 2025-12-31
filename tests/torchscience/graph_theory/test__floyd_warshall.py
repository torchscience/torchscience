# tests/torchscience/graph_theory/test__floyd_warshall.py

import torch


def test_import_floyd_warshall():
    """Can import floyd_warshall from graph_theory module."""
    from torchscience.graph_theory import floyd_warshall

    assert callable(floyd_warshall)


def test_cpp_operator_registered():
    """C++ operator is registered with torch.ops."""
    import torchscience._csrc  # noqa: F401

    assert hasattr(torch.ops.torchscience, "floyd_warshall")
