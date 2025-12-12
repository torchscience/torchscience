import torch
from torch import Tensor


def exponential_integral_e_1(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Exponential integral E_1.

    .. math::

        E_1(x) = \\int_x^\\infty \\frac{e^{-t}}{t} dt

    This is a special case of the generalized exponential integral E_n(x) with n=1.

    Parameters
    ----------
    input : Tensor, shape=(...)
        Input tensor. Values must be positive.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The exponential integral E_1 evaluated at each element of `input`.

    Examples
    --------
    >>> exponential_integral_e_1(torch.tensor([0.5, 1.0, 2.0]))
    tensor([0.5598, 0.2194, 0.0489])

    Notes
    -----
    The exponential integral E_1 is related to the exponential integral Ei by:

    .. math::

        E_1(x) = -\\text{Ei}(-x)

    for x > 0.
    """
    output: Tensor = torch.ops.torchscience._exponential_integral_e_1(input)

    if out is not None:
        out.copy_(output)

        return out

    return output
