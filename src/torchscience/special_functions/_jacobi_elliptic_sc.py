import torch
from torch import Tensor


def jacobi_elliptic_sc(u: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Jacobi elliptic function sc.

    .. math::

        \\text{sc}(u, k) = \\frac{\\text{sn}(u, k)}{\\text{cn}(u, k)} = \\tan(\\text{am}(u, k))

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
        The Jacobi elliptic function sc evaluated at each element pair.

    Notes
    -----
    The derivative with respect to u is:

    .. math::

        \\frac{d}{du} \\text{sc}(u, k) = \\frac{\\text{dn}(u, k)}{\\text{cn}^2(u, k)}

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    Examples
    --------
    >>> jacobi_elliptic_sc(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([0.])

    >>> jacobi_elliptic_sc(torch.tensor([1.0]), torch.tensor([0.0]))
    tensor([1.5574])
    """
    output: Tensor = torch.ops.torchscience._jacobi_elliptic_sc(u, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
