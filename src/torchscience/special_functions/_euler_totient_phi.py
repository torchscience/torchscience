import torch
from torch import Tensor


def euler_totient_phi(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Euler's totient function :math:`\phi(n)`.

    Counts the number of integers from 1 to n that are coprime to n:

    .. math::

        \phi(n) = |\{k : 1 \leq k \leq n, \gcd(k, n) = 1\}|

    Properties:

    - :math:`\phi(1) = 1`
    - :math:`\phi(p) = p - 1` for prime p
    - :math:`\phi(p^k) = p^{k-1}(p - 1)` for prime power
    - Multiplicative: :math:`\phi(mn) = \phi(m)\phi(n)` when :math:`\gcd(m,n) = 1`

    Parameters
    ----------
    input : Tensor
        Input tensor containing positive integers n.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Euler totient function :math:`\phi(n)` for each element.

    Examples
    --------
    >>> euler_totient_phi(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([1., 1., 2., 2., 4., 2.])
    """
    output: Tensor = torch.ops.torchscience._euler_totient_phi(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
