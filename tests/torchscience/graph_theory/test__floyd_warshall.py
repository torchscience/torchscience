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


def test_meta_shape_inference():
    """Meta backend correctly infers output shapes."""
    import torchscience._csrc  # noqa: F401

    input_meta = torch.empty(5, 5, device="meta")
    dist, pred, _ = torch.ops.torchscience.floyd_warshall(input_meta, True)

    assert dist.shape == (5, 5)
    assert pred.shape == (5, 5)
    assert pred.dtype == torch.int64


def test_cpu_simple_graph():
    """CPU backend computes correct shortest paths for simple graph."""
    import torchscience._csrc  # noqa: F401

    # Simple 3-node graph:
    #   0 --1--> 1 --1--> 2
    #   0 --3--> 2 (direct but longer)
    inf = float("inf")
    adj = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ]
    )

    dist, pred, has_neg = torch.ops.torchscience.floyd_warshall(adj, True)

    assert not has_neg
    expected_dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],  # 0->2 via 1 is shorter (1+1=2 < 3)
            [inf, 0.0, 1.0],
            [inf, inf, 0.0],
        ]
    )
    assert torch.allclose(dist, expected_dist)
