import torch
from torch import Tensor


def inverse_jacobi_elliptic_sc(x: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Inverse Jacobi elliptic function sc (arcsc).

    Parameters
    ----------
    x : Tensor, shape=(...)
        Input values.

    k : Tensor, shape=(...)
        Elliptic modulus. Must satisfy :math:`0 \\leq k \\leq 1`.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The inverse Jacobi elliptic function sc evaluated at each element pair.

    Notes
    -----
    The derivative with respect to x is:

    .. math::

        \\frac{d}{dx} \\text{arcsc}(x, k) = \\frac{1}{\\sqrt{(1 + x^2)(1 + (1-k^2) x^2)}}

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    The function satisfies:

    .. math::

        \\text{sc}(\\text{arcsc}(x, k), k) = x

    Examples
    --------
    >>> inverse_jacobi_elliptic_sc(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([0.])
    """
    output: Tensor = torch.ops.torchscience._inverse_jacobi_elliptic_sc(x, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
