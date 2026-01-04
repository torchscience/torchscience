"""Frequency transform functions for analog filters."""

from typing import Tuple, Union

import torch
from torch import Tensor


def lp2lp_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a different cutoff frequency.

    Performs the analog frequency scaling s -> s/wo, which scales the
    cutoff frequency from 1 rad/s to wo rad/s.

    Parameters
    ----------
    z : Tensor
        Zeros of the analog filter.
    p : Tensor
        Poles of the analog filter.
    k : Tensor
        System gain of the analog filter.
    wo : float or Tensor
        New cutoff frequency (rad/s).

    Returns
    -------
    z : Tensor
        Zeros of the transformed filter.
    p : Tensor
        Poles of the transformed filter.
    k : Tensor
        System gain of the transformed filter.

    Notes
    -----
    The transformation s -> s/wo scales all poles and zeros by wo,
    and adjusts the gain to maintain the correct DC response.

    The gain is multiplied by wo^(len(p) - len(z)) to account for
    the degree difference between numerator and denominator.
    """
    if not isinstance(wo, Tensor):
        wo = torch.as_tensor(wo, dtype=k.dtype, device=k.device)

    # Scale zeros and poles by wo
    z_new = z * wo
    p_new = p * wo

    # Adjust gain: k * wo^(degree difference)
    degree_diff = p.numel() - z.numel()
    k_new = k * (wo**degree_diff)

    return z_new, p_new, k_new


def lp2hp_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a highpass filter.

    Performs the analog transformation s -> wo/s, which converts a
    lowpass filter with cutoff 1 rad/s to a highpass filter with
    cutoff wo rad/s.

    Parameters
    ----------
    z : Tensor
        Zeros of the analog lowpass filter.
    p : Tensor
        Poles of the analog lowpass filter.
    k : Tensor
        System gain of the analog lowpass filter.
    wo : float or Tensor
        Cutoff frequency of the highpass filter (rad/s).

    Returns
    -------
    z : Tensor
        Zeros of the highpass filter.
    p : Tensor
        Poles of the highpass filter.
    k : Tensor
        System gain of the highpass filter.

    Notes
    -----
    The transformation s -> wo/s:
    - Maps poles p_k to wo/p_k
    - Maps zeros z_k to wo/z_k
    - Adds (len(p) - len(z)) zeros at s=0
    - Adjusts gain to maintain correct high-frequency response
    """
    if not isinstance(wo, Tensor):
        wo = torch.as_tensor(wo, dtype=k.dtype, device=k.device)

    degree_diff = p.numel() - z.numel()

    # Transform poles: p_new = wo / p
    p_new = wo / p

    # Transform existing zeros and add zeros at origin
    if z.numel() > 0:
        z_transformed = wo / z
    else:
        z_transformed = torch.empty(0, dtype=p.dtype, device=p.device)

    # Add zeros at origin to match degree difference
    zeros_at_origin = torch.zeros(degree_diff, dtype=p.dtype, device=p.device)
    z_new = torch.cat([z_transformed, zeros_at_origin])

    # Adjust gain using ORIGINAL z and p (not transformed)
    # k_new = k * real(prod(-z) / prod(-p))
    # For Butterworth (no zeros): prod(-z) = 1
    if z.numel() > 0:
        prod_neg_z = torch.prod(-z)
    else:
        prod_neg_z = torch.tensor(1.0, dtype=p.dtype, device=p.device)
    prod_neg_p = torch.prod(-p)
    k_new = k * torch.real(prod_neg_z / prod_neg_p)

    return z_new, p_new, k_new
