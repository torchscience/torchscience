from torch import Tensor

import torch


def hankel_h_2(nu: Tensor, x: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Hankel function of the second kind.

    .. math::
        H^{(2)}_\nu(x) = J_\nu(x) - i Y_\nu(x)

    where :math:`J_\nu` is the Bessel function of the first kind and
    :math:`Y_\nu` is the Bessel function of the second kind.

    For real-valued inputs, returns the magnitude :math:`|H^{(2)}_\nu(x)|`.
    For complex-valued inputs, returns the full complex Hankel function.

    Parameters
    ----------
    nu : Tensor
        Order of the Hankel function.
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hankel function of the second kind.
    """
    output = torch.ops.torchscience._hankel_h_2(nu, x)

    if out is not None:
        out.copy_(output)
        return out

    return output
