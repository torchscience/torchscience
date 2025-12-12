import torch
from torch import Tensor


def bessel_y(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Bessel function of the second kind.

    .. math::

        Y_{\\nu}(x) = \\frac{J_{\\nu}(x) \\cos(\\nu \\pi) - J_{-\\nu}(x)}{\\sin(\\nu \\pi)}

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
        The Bessel function of the second kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `nu` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} Y_{\\nu}(x) = \\frac{1}{2}\\left(Y_{\\nu-1}(x) - Y_{\\nu+1}(x)\\right)

    Examples
    --------
    >>> bessel_y(torch.tensor([0.0]), torch.tensor([1.0]))
    tensor([0.0883])

    >>> bessel_y(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([ 0.0883, -0.7812, -1.6507])
    """
    output: Tensor = torch.ops.torchscience._bessel_y(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
