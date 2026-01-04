"""Quaternion representation and operations."""

from __future__ import annotations

from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class Quaternion:
    """Unit quaternion representing a 3D rotation.

    Uses scalar-first (wxyz) convention: q = w + xi + yj + zk.

    Attributes
    ----------
    wxyz : Tensor
        Quaternion components in [w, x, y, z] order, shape (..., 4).
        For unit quaternions: w^2 + x^2 + y^2 + z^2 = 1.

    Examples
    --------
    Identity rotation:
        Quaternion(wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]))

    90-degree rotation around z-axis:
        Quaternion(wxyz=torch.tensor([0.7071, 0.0, 0.0, 0.7071]))

    Batch of quaternions:
        Quaternion(wxyz=torch.randn(100, 4))
    """

    wxyz: Tensor


def quaternion(wxyz: Tensor) -> Quaternion:
    """Create quaternion from wxyz tensor.

    Parameters
    ----------
    wxyz : Tensor
        Quaternion components [w, x, y, z], shape (..., 4).

    Returns
    -------
    Quaternion
        Quaternion instance.

    Raises
    ------
    ValueError
        If wxyz does not have last dimension 4.

    Examples
    --------
    >>> q = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    >>> q.wxyz
    tensor([1., 0., 0., 0.])
    """
    if wxyz.shape[-1] != 4:
        raise ValueError(
            f"quaternion: wxyz must have last dimension 4, got {wxyz.shape[-1]}"
        )
    return Quaternion(wxyz=wxyz)
