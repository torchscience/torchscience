import torch
from torch import Tensor


def spherical_bessel_j(n: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Spherical Bessel function of the first kind.

    .. math::

        j_n(x) = \\sqrt{\\frac{\\pi}{2x}} J_{n+1/2}(x)

    Parameters
    ----------
    n : Tensor, shape=(...)
        Order of the spherical Bessel function.

    x : Tensor, shape=(...)
        Argument of the spherical Bessel function.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The spherical Bessel function of the first kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `n` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} j_n(x) = j_{n-1}(x) - \\frac{n+1}{x} j_n(x)

    Examples
    --------
    >>> spherical_bessel_j(torch.tensor([0.0]), torch.tensor([1.0]))
    tensor([0.8415])

    >>> spherical_bessel_j(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([0.8415, 0.3012, 0.0621])
    """
    output: Tensor = torch.ops.torchscience._spherical_bessel_j(n, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
