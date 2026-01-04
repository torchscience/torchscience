"""Quaternion representation and operations."""

from __future__ import annotations

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


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


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions using the Hamilton product.

    .. math::

        (q_1 \\cdot q_2)_w = w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2

        (q_1 \\cdot q_2)_x = w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2

        (q_1 \\cdot q_2)_y = w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2

        (q_1 \\cdot q_2)_z = w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2

    Where :math:`q_1 = w_1 + x_1 i + y_1 j + z_1 k` and
    :math:`q_2 = w_2 + x_2 i + y_2 j + z_2 k`.

    Parameters
    ----------
    q1 : Quaternion
        First quaternion, shape (..., 4).
    q2 : Quaternion
        Second quaternion, shape (..., 4). Batch dimensions broadcast with q1.

    Returns
    -------
    Quaternion
        Product q1 * q2, shape is broadcast of q1 and q2 batch dimensions.

    Notes
    -----
    - Quaternion multiplication is **non-commutative**: q1 * q2 != q2 * q1.
    - For unit quaternions representing rotations, q1 * q2 represents
      rotation q1 followed by rotation q2.
    - Works for any quaternions, not just unit quaternions.

    See Also
    --------
    quaternion_inverse : Compute quaternion inverse.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    Examples
    --------
    >>> q1 = quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # identity
    >>> q2 = quaternion(torch.tensor([0.7071, 0.7071, 0.0, 0.0]))  # 90deg around x
    >>> quaternion_multiply(q1, q2).wxyz
    tensor([0.7071, 0.7071, 0.0000, 0.0000])
    """
    result = torch.ops.torchscience.quaternion_multiply(q1.wxyz, q2.wxyz)
    return Quaternion(wxyz=result)


def quaternion_inverse(q: Quaternion) -> Quaternion:
    """Compute inverse of a unit quaternion.

    For unit quaternions, the inverse equals the conjugate:

    .. math::

        q^{-1} = [w, -x, -y, -z]

    Where :math:`q = w + xi + yj + zk` is a unit quaternion with
    :math:`|q| = w^2 + x^2 + y^2 + z^2 = 1`.

    Parameters
    ----------
    q : Quaternion
        Unit quaternion, shape (..., 4).

    Returns
    -------
    Quaternion
        Inverse quaternion q^(-1), shape (..., 4).

    Notes
    -----
    - Only valid for unit quaternions (|q| = 1).
    - For general quaternions, use q* / |q|^2.
    - The inverse satisfies q * q^(-1) = q^(-1) * q = identity.

    See Also
    --------
    quaternion_multiply : Multiply two quaternions.
    quaternion_normalize : Normalize to unit quaternion.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal

    Examples
    --------
    >>> q = quaternion(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    >>> q_inv = quaternion_inverse(q)
    >>> q_inv.wxyz
    tensor([ 0.5000, -0.5000, -0.5000, -0.5000])

    Verify that q * q^(-1) gives identity:

    >>> quaternion_multiply(q, q_inv).wxyz
    tensor([1., 0., 0., 0.])
    """
    result = torch.ops.torchscience.quaternion_inverse(q.wxyz)
    return Quaternion(wxyz=result)


def quaternion_normalize(q: Quaternion) -> Quaternion:
    """Normalize a quaternion to unit length.

    Computes the unit quaternion by dividing by its norm:

    .. math::

        \\hat{q} = \\frac{q}{\\|q\\|}

    Where :math:`\\|q\\| = \\sqrt{w^2 + x^2 + y^2 + z^2}` is the quaternion norm.

    Parameters
    ----------
    q : Quaternion
        Input quaternion, shape (..., 4).

    Returns
    -------
    Quaternion
        Unit quaternion with ||q|| = 1, shape (..., 4).

    Notes
    -----
    - The output quaternion has unit norm: ||output|| = 1.
    - Normalizing an already-normalized quaternion returns the same quaternion
      (up to numerical precision).
    - For zero quaternions, the result is undefined (division by zero).
    - Useful for ensuring quaternions remain valid rotations after
      numerical operations that may cause drift from unit norm.

    See Also
    --------
    quaternion_inverse : Compute quaternion inverse.
    quaternion_multiply : Multiply two quaternions.

    References
    ----------
    .. [1] Hamilton, W.R. "Elements of Quaternions", 1866.
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Unit_quaternion

    Examples
    --------
    >>> q = quaternion(torch.tensor([2.0, 0.0, 0.0, 0.0]))
    >>> q_norm = quaternion_normalize(q)
    >>> q_norm.wxyz
    tensor([1., 0., 0., 0.])

    Normalize a general quaternion:

    >>> q = quaternion(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    >>> q_norm = quaternion_normalize(q)
    >>> torch.linalg.norm(q_norm.wxyz)
    tensor(1.)
    """
    result = torch.ops.torchscience.quaternion_normalize(q.wxyz)
    return Quaternion(wxyz=result)
