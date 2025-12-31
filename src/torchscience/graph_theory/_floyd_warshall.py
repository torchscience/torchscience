"""Floyd-Warshall all-pairs shortest paths implementation."""

from torch import Tensor


class NegativeCycleError(ValueError):
    """Raised when the graph contains a negative cycle."""

    pass


def floyd_warshall(
    input: Tensor,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Compute all-pairs shortest paths using the Floyd-Warshall algorithm.

    Args:
        input: Adjacency matrix of shape (*, N, N) where input[..., i, j]
               is the edge weight from node i to j. Use float('inf') for
               missing edges. Can be dense or sparse COO tensor.
        directed: If True, treat graph as directed. If False, symmetrize
                  the adjacency matrix by taking element-wise minimum.

    Returns:
        distances: Tensor of shape (*, N, N) with shortest path distances.
        predecessors: Tensor of shape (*, N, N) with dtype int64.
                      predecessors[..., i, j] is the node before j on the
                      shortest path from i to j, or -1 if no path exists.

    Raises:
        NegativeCycleError: If the graph contains a negative cycle.
    """
    raise NotImplementedError("floyd_warshall is not yet implemented")
