import torch
from torch import Tensor


def modified_bessel_i(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Modified Bessel function of the first kind.

    .. math::

        I_{\\nu}(x) = \\sum_{m=0}^{\\infty} \\frac{1}{m! \\Gamma(m + \\nu + 1)}
                      \\left(\\frac{x}{2}\\right)^{2m + \\nu}

    Parameters
    ----------
    nu : Tensor, shape=(...)
        Order of the modified Bessel function.

    x : Tensor, shape=(...)
        Argument of the modified Bessel function.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The modified Bessel function of the first kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `nu` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} I_{\\nu}(x) = \\frac{1}{2}\\left(I_{\\nu-1}(x) + I_{\\nu+1}(x)\\right)

    Examples
    --------
    >>> modified_bessel_i(torch.tensor([0.0]), torch.tensor([0.0]))
    tensor([1.])

    >>> modified_bessel_i(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([1.2661, 0.5652, 0.1358])
    """
    output: Tensor = torch.ops.torchscience._modified_bessel_i(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
