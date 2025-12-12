import torch
from torch import Tensor


def bessel_j(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Bessel function of the first kind.

    .. math::

        J_{\\nu}(x) = \\sum_{m=0}^{\\infty} \\frac{(-1)^m}{m! \\Gamma(m + \\nu + 1)}
                      \\left(\\frac{x}{2}\\right)^{2m + \\nu}

    Parameters
    ----------
    nu : Tensor, shape=(...)
        Order of the Bessel function.

    x : Tensor, shape=(...)
        Argument of the Bessel function.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Bessel function of the first kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `nu` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} J_{\\nu}(x) = \\frac{1}{2}\\left(J_{\\nu-1}(x) - J_{\\nu+1}(x)\\right)

    Examples
    --------
    >>> bessel_j(torch.tensor([0.0]), torch.tensor([0.0]))
    tensor([1.])

    >>> bessel_j(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([0.7652, 0.4401, 0.1149])
    """
    output: Tensor = torch.ops.torchscience._bessel_j(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
