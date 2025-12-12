from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["stirling_number_s_1"]


def stirling_number_s_1(n: Tensor, k: Tensor) -> Tensor:
    r"""
    Unsigned Stirling numbers of the first kind :math:`|s(n, k)|`.

    The unsigned Stirling numbers of the first kind count the number of
    permutations of :math:`n` elements with exactly :math:`k` disjoint cycles.

    They satisfy the recurrence relation:

    .. math::
        |s(n+1, k)| = n \cdot |s(n, k)| + |s(n, k-1)|

    with :math:`|s(0, 0)| = 1` and :math:`|s(n, 0)| = |s(0, k)| = 0` for
    :math:`n, k > 0`.

    Parameters
    ----------
    n : Tensor
        Non-negative integer parameter (total number of elements).
    k : Tensor
        Non-negative integer parameter (number of cycles), with :math:`k \leq n`.

    Returns
    -------
    Tensor
        The unsigned Stirling number of the first kind :math:`|s(n, k)|`.
    """
    return torchscience.ops.torchscience._stirling_number_s_1(n, k)
