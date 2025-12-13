import torch
from torch import Tensor


def liouville_lambda(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Liouville function :math:`\lambda(n)`.

    The Liouville function is a completely multiplicative function defined as:

    .. math::
        \lambda(n) = (-1)^{\Omega(n)}

    where :math:`\Omega(n)` is the number of prime factors of n counted with
    multiplicity.

    Properties:

    - Completely multiplicative: :math:`\lambda(mn) = \lambda(m)\lambda(n)`
    - :math:`\lambda(1) = 1`
    - :math:`\lambda(p) = -1` for any prime p
    - :math:`\lambda(p^k) = (-1)^k`
    - Related to Mobius function: :math:`|\mu(n)| = \lambda(n) \mu(n)^2`
    - :math:`\sum_{d | n} \lambda(d) = \begin{cases} 1 & \text{if } n \text{ is a perfect square} \\ 0 & \text{otherwise} \end{cases}`

    Parameters
    ----------
    input : Tensor
        Input tensor containing positive integers n.

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The Liouville function :math:`\lambda(n)` for each element.

    Examples
    --------
    >>> liouville_lambda(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    tensor([ 1., -1., -1.,  1., -1.,  1.])
    """
    output: Tensor = torch.ops.torchscience._liouville_lambda(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
