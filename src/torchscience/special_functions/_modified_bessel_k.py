import torch
from torch import Tensor


def modified_bessel_k(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Modified Bessel function of the second kind.

    .. math::

        K_{\\nu}(x) = \\frac{\\pi}{2} \\frac{I_{-\\nu}(x) - I_{\\nu}(x)}{\\sin(\\nu \\pi)}

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
        The modified Bessel function of the second kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `nu` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} K_{\\nu}(x) = -\\frac{1}{2}\\left(K_{\\nu-1}(x) + K_{\\nu+1}(x)\\right)

    Examples
    --------
    >>> modified_bessel_k(torch.tensor([0.0]), torch.tensor([1.0]))
    tensor([0.4210])

    >>> modified_bessel_k(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([0.4210, 0.6019, 1.6248])
    """
    output: Tensor = torch.ops.torchscience._modified_bessel_k(nu, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
