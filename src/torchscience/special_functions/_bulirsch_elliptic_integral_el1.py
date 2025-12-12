import torch
from torch import Tensor


def bulirsch_elliptic_integral_el1(x: Tensor, kc: Tensor, *, out: Tensor | None = None) -> Tensor:
    r"""
    Bulirsch's incomplete elliptic integral of the first kind.

    .. math::
        \text{el1}(x, k_c) = x \cdot R_F(1, 1 + k_c^2 x^2, 1 + x^2)

    where :math:`R_F` is Carlson's elliptic integral and :math:`k_c` is
    the complementary modulus.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kc : Tensor
        Complementary modulus.
    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        Bulirsch's incomplete elliptic integral of the first kind.
    """
    output: Tensor = torch.ops.torchscience._bulirsch_elliptic_integral_el1(x, kc)

    if out is not None:
        out.copy_(output)
        return out

    return output
