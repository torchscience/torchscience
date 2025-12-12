from torch import Tensor

import torchscience.ops.torchscience

__all__ = ["stirling_number_s_2"]


def stirling_number_s_2(n: Tensor, k: Tensor) -> Tensor:
    r"""
    Stirling numbers of the second kind :math:`S(n, k)`.

    The Stirling numbers of the second kind count the number of ways to
    partition a set of :math:`n` elements into exactly :math:`k` non-empty
    subsets.

    They satisfy the recurrence relation:

    .. math::
        S(n+1, k) = k \cdot S(n, k) + S(n, k-1)

    with :math:`S(0, 0) = 1` and :math:`S(n, 0) = S(0, k) = 0` for
    :math:`n, k > 0`.

    Parameters
    ----------
    n : Tensor
        Non-negative integer parameter (total number of elements).
    k : Tensor
        Non-negative integer parameter (number of subsets), with :math:`k \leq n`.

    Returns
    -------
    Tensor
        The Stirling number of the second kind :math:`S(n, k)`.
    """
    return torchscience.ops.torchscience._stirling_number_s_2(n, k)
