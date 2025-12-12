import torch
from torch import Tensor


def jacobi_elliptic_sn(u: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Jacobi elliptic function sn.

    .. math::

        \\text{sn}(u, k) = \\sin(\\text{am}(u, k))

    where :math:`\\text{am}(u, k)` is the Jacobi amplitude function.

    Parameters
    ----------
    u : Tensor, shape=(...)
        Argument of the elliptic function.

    k : Tensor, shape=(...)
        Elliptic modulus. Must satisfy :math:`0 \\leq k \\leq 1`.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Jacobi elliptic function sn evaluated at each element pair.

    Notes
    -----
    The derivative with respect to u is:

    .. math::

        \\frac{d}{du} \\text{sn}(u, k) = \\text{cn}(u, k) \\cdot \\text{dn}(u, k)

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    The function satisfies:

    .. math::

        \\text{sn}^2(u, k) + \\text{cn}^2(u, k) = 1

    Examples
    --------
    >>> jacobi_elliptic_sn(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([0.])

    >>> jacobi_elliptic_sn(torch.tensor([1.0]), torch.tensor([0.0]))
    tensor([0.8415])
    """
    output: Tensor = torch.ops.torchscience._jacobi_elliptic_sn(u, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
