import torch
from torch import Tensor


def spherical_modified_bessel_i(n: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Spherical modified Bessel function of the first kind.

    .. math::

        i_n(x) = \\sqrt{\\frac{\\pi}{2x}} I_{n+1/2}(x)

    Parameters
    ----------
    n : Tensor, shape=(...)
        Order of the spherical modified Bessel function.

    x : Tensor, shape=(...)
        Argument of the spherical modified Bessel function.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The spherical modified Bessel function of the first kind evaluated at each element pair.

    Notes
    -----
    Gradients with respect to the order `n` are not supported and will be zero.
    Gradients with respect to `x` use the recurrence relation:

    .. math::

        \\frac{d}{dx} i_n(x) = i_{n-1}(x) - \\frac{n+1}{x} i_n(x)

    Examples
    --------
    >>> spherical_modified_bessel_i(torch.tensor([0.0]), torch.tensor([1.0]))
    tensor([1.1752])

    >>> spherical_modified_bessel_i(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    tensor([1.1752, 0.3679, 0.0717])
    """
    output: Tensor = torch.ops.torchscience._spherical_modified_bessel_i(n, x)

    if out is not None:
        out.copy_(output)

        return out

    return output
