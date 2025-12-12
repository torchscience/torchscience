import torch
from torch import Tensor


def inverse_jacobi_elliptic_sn(x: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Inverse Jacobi elliptic function sn (arcsn).

    .. math::

        \\text{arcsn}(x, k) = F(\\arcsin(x), k)

    where :math:`F(\\phi, k)` is the incomplete elliptic integral of the first kind.

    Parameters
    ----------
    x : Tensor, shape=(...)
        Input values. Must satisfy :math:`-1 \\leq x \\leq 1`.

    k : Tensor, shape=(...)
        Elliptic modulus. Must satisfy :math:`0 \\leq k \\leq 1`.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The inverse Jacobi elliptic function sn evaluated at each element pair.

    Notes
    -----
    The derivative with respect to x is:

    .. math::

        \\frac{d}{dx} \\text{arcsn}(x, k) = \\frac{1}{\\sqrt{(1 - x^2)(1 - k^2 x^2)}}

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    The function satisfies:

    .. math::

        \\text{sn}(\\text{arcsn}(x, k), k) = x

    Examples
    --------
    >>> inverse_jacobi_elliptic_sn(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([0.])

    >>> inverse_jacobi_elliptic_sn(torch.tensor([0.5]), torch.tensor([0.0]))
    tensor([0.5236])
    """
    output: Tensor = torch.ops.torchscience._inverse_jacobi_elliptic_sn(x, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
