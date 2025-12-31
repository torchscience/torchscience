# tests/torchscience/graph_theory/test__floyd_warshall.py


def test_import_floyd_warshall():
    """Can import floyd_warshall from graph_theory module."""
    from torchscience.graph_theory import floyd_warshall

    assert callable(floyd_warshall)
