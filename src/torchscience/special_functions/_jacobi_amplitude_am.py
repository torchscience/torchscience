import torch
from torch import Tensor


def jacobi_amplitude_am(u: Tensor, k: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Jacobi amplitude function.

    .. math::

        \\text{am}(u, k) = \\phi

    where :math:`\\phi` is defined implicitly by the incomplete elliptic
    integral of the first kind:

    .. math::

        u = F(\\phi, k) = \\int_0^{\\phi} \\frac{d\\theta}{\\sqrt{1 - k^2 \\sin^2 \\theta}}

    The Jacobi amplitude is the inverse of the incomplete elliptic integral
    of the first kind.

    Parameters
    ----------
    u : Tensor, shape=(...)
        Argument of the amplitude function.

    k : Tensor, shape=(...)
        Elliptic modulus. Must satisfy :math:`0 \\leq k \\leq 1`.

    out : Tensor, shape=(...), optional
        Output tensor.

    Returns
    -------
    Tensor, shape=(...)
        The Jacobi amplitude evaluated at each element pair.

    Notes
    -----
    The Jacobi elliptic functions are related to the amplitude by:

    .. math::

        \\text{sn}(u, k) &= \\sin(\\text{am}(u, k)) \\\\
        \\text{cn}(u, k) &= \\cos(\\text{am}(u, k)) \\\\
        \\text{dn}(u, k) &= \\sqrt{1 - k^2 \\sin^2(\\text{am}(u, k))}

    Gradients with respect to the modulus `k` are not supported and will
    be zero. Gradients with respect to `u` use:

    .. math::

        \\frac{d}{du} \\text{am}(u, k) = \\text{dn}(u, k)

    Examples
    --------
    >>> jacobi_amplitude_am(torch.tensor([0.0]), torch.tensor([0.5]))
    tensor([0.])

    >>> jacobi_amplitude_am(torch.tensor([1.0]), torch.tensor([0.0]))
    tensor([1.])
    """
    output: Tensor = torch.ops.torchscience._jacobi_amplitude_am(u, k)

    if out is not None:
        out.copy_(output)

        return out

    return output
