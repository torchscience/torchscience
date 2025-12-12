import torch
from torch import Tensor


def jacobi_elliptic_cd(u: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Jacobi elliptic function cd.

    .. math::

        \\text{cd}(u, k) = \\frac{\\text{cn}(u, k)}{\\text{dn}(u, k)}

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
        The Jacobi elliptic function cd evaluated at each element pair.

    Notes
    -----
    The derivative with respect to u is:

    .. math::

        \\frac{d}{du} \\text{cd}(u, k) = -\\frac{(1 - k^2) \\text{sn}(u, k)}{\\text{dn}^2(u, k)}

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    Examples
    --------
    >>> jacobi_elliptic_cd(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([1.])

    >>> jacobi_elliptic_cd(torch.tensor([1.0]), torch.tensor([0.0]))
    tensor([0.5403])
    """
    output: Tensor = torch.ops.torchscience._jacobi_elliptic_cd(u, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
