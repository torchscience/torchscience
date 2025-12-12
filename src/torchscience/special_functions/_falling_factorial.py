import torch
from torch import Tensor


def falling_factorial(x: Tensor, n: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Computes the falling factorial of x with n terms.

    .. math::

        \\text{falling\\_factorial}(x, n) = x^{\\underline{n}} = x \\cdot (x-1) \\cdot (x-2) \\cdots (x-n+1) = \\frac{\\Gamma(x+1)}{\\Gamma(x-n+1)}

    Also known as the descending factorial or falling sequential product.

    Parameters
    ----------
    x : Tensor
        Base value.

    n : Tensor
        Number of terms (non-negative integer).

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        The falling factorial of x with n terms.

    Examples
    --------
    >>> falling_factorial(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([3.0, 3.0, 3.0]))
    tensor([ 60., 120., 210.])

    >>> # (5)_3 = 5 * 4 * 3 = 60
    >>> # (6)_3 = 6 * 5 * 4 = 120
    >>> # (7)_3 = 7 * 6 * 5 = 210
    """
    output: Tensor = torch.ops.torchscience._falling_factorial(x, n)

    if out is not None:
        out.copy_(output)

        return out

    return output
