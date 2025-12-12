import torch
from torch import Tensor


def exponential_integral_e(n: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Exponential integral E_n.

    .. math::

        E_n(x) = \\int_1^{\\infty} \\frac{e^{-xt}}{t^n} dt

    Parameters
    ----------
    n : Tensor, shape=(...)
        Order of the exponential integral.

    x : Tensor, shape=(...)
        Argument of the exponential integral.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The exponential integral E_n evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `n` are not supported and will be zero.
    Gradients with respect to `x` use the relation:

    .. math::

        \\frac{d}{dx} E_n(x) = -E_{n-1}(x)

    Examples
    --------
    >>> exponential_integral_e(torch.tensor([1.0]), torch.tensor([1.0]))
    tensor([0.2194])

    >>> exponential_integral_e(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([0.2194, 0.1485, 0.1097])
    """
    output: Tensor = torch.ops.torchscience._exponential_integral_e(n, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
