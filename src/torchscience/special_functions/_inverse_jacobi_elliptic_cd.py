import torch
from torch import Tensor


def inverse_jacobi_elliptic_cd(x: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Inverse Jacobi elliptic function cd (arccd).

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
        The inverse Jacobi elliptic function cd evaluated at each element pair.

    Notes
    -----
    The derivative with respect to x is:

    .. math::

        \\frac{d}{dx} \\text{arccd}(x, k) = -\\frac{1}{\\sqrt{(1 - x^2)(1 - k^2 x^2)}}

    Gradients with respect to the modulus `k` are not supported and will
    be zero.

    The function satisfies:

    .. math::

        \\text{cd}(\\text{arccd}(x, k), k) = x

    Examples
    --------
    >>> inverse_jacobi_elliptic_cd(torch.tensor([1.0]), torch.tensor([0.5]))
    tensor([0.])
    """
    output: Tensor = torch.ops.torchscience._inverse_jacobi_elliptic_cd(x, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
