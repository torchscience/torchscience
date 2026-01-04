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


def lp2bp_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: Union[float, Tensor] = 1.0,
    bw: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a bandpass filter.

    Performs the analog transformation s -> (s^2 + wo^2) / (bw * s), which
    converts a lowpass filter with cutoff 1 rad/s to a bandpass filter with
    center frequency wo rad/s and bandwidth bw rad/s.

    Parameters
    ----------
    z : Tensor
        Zeros of the analog lowpass filter.
    p : Tensor
        Poles of the analog lowpass filter.
    k : Tensor
        System gain of the analog lowpass filter.
    wo : float or Tensor
        Center frequency of the bandpass filter (rad/s).
    bw : float or Tensor
        Bandwidth of the bandpass filter (rad/s).

    Returns
    -------
    z : Tensor
        Zeros of the bandpass filter.
    p : Tensor
        Poles of the bandpass filter.
    k : Tensor
        System gain of the bandpass filter.

    Notes
    -----
    The transformation s -> (s^2 + wo^2) / (bw * s):
    - Doubles the filter order (each pole becomes two poles)
    - Adds (len(p) - len(z)) zeros at s=0
    - Maps the lowpass cutoff to the bandpass edges

    For each pole p_k, the new poles are:
        p_new = (bw * p_k / 2) ± sqrt((bw * p_k / 2)^2 - wo^2)
    """
    if not isinstance(wo, Tensor):
        wo = torch.as_tensor(wo, dtype=k.dtype, device=k.device)
    if not isinstance(bw, Tensor):
        bw = torch.as_tensor(bw, dtype=k.dtype, device=k.device)

    degree_diff = p.numel() - z.numel()
    wo_sq = wo * wo

    # Transform poles: each pole becomes two poles
    # p_new = (bw * p / 2) ± sqrt((bw * p / 2)^2 - wo^2)
    half_bw_p = (bw * p) / 2
    discriminant = half_bw_p * half_bw_p - wo_sq
    sqrt_disc = torch.sqrt(discriminant.to(p.dtype))
    p_new_1 = half_bw_p + sqrt_disc
    p_new_2 = half_bw_p - sqrt_disc
    p_new = torch.cat([p_new_1, p_new_2])

    # Transform zeros (if any) and add zeros at origin
    if z.numel() > 0:
        half_bw_z = (bw * z) / 2
        disc_z = half_bw_z * half_bw_z - wo_sq
        sqrt_disc_z = torch.sqrt(disc_z.to(z.dtype))
        z_new_1 = half_bw_z + sqrt_disc_z
        z_new_2 = half_bw_z - sqrt_disc_z
        z_transformed = torch.cat([z_new_1, z_new_2])
    else:
        z_transformed = torch.empty(0, dtype=p.dtype, device=p.device)

    # Add zeros at origin
    zeros_at_origin = torch.zeros(degree_diff, dtype=p.dtype, device=p.device)
    z_new = torch.cat([z_transformed, zeros_at_origin])

    # Adjust gain: k * bw^(degree_diff)
    k_new = k * (bw**degree_diff)

    return z_new, p_new, k_new


def lp2bs_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: Union[float, Tensor] = 1.0,
    bw: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform a lowpass filter to a bandstop (notch) filter.

    Performs the analog transformation s -> bw * s / (s^2 + wo^2), which
    converts a lowpass filter with cutoff 1 rad/s to a bandstop filter with
    center frequency wo rad/s and bandwidth bw rad/s.

    Parameters
    ----------
    z : Tensor
        Zeros of the analog lowpass filter.
    p : Tensor
        Poles of the analog lowpass filter.
    k : Tensor
        System gain of the analog lowpass filter.
    wo : float or Tensor
        Center frequency of the bandstop filter (rad/s).
    bw : float or Tensor
        Bandwidth of the bandstop filter (rad/s).

    Returns
    -------
    z : Tensor
        Zeros of the bandstop filter.
    p : Tensor
        Poles of the bandstop filter.
    k : Tensor
        System gain of the bandstop filter.

    Notes
    -----
    The transformation s -> bw * s / (s^2 + wo^2):
    - Doubles the filter order (each pole becomes two poles)
    - Adds 2 * (len(p) - len(z)) zeros at ±j*wo
    - Creates a notch at the center frequency

    For each pole p_k, the new poles are:
        p_new = (bw / (2 * p_k)) ± sqrt((bw / (2 * p_k))^2 - wo^2)
    """
    if not isinstance(wo, Tensor):
        wo = torch.as_tensor(wo, dtype=k.dtype, device=k.device)
    if not isinstance(bw, Tensor):
        bw = torch.as_tensor(bw, dtype=k.dtype, device=k.device)

    degree_diff = p.numel() - z.numel()
    wo_sq = wo * wo

    # Transform poles: each pole becomes two poles
    # p_new = (bw / (2 * p)) ± sqrt((bw / (2 * p))^2 - wo^2)
    half_bw_over_p = bw / (2 * p)
    discriminant = half_bw_over_p * half_bw_over_p - wo_sq
    sqrt_disc = torch.sqrt(discriminant.to(p.dtype))
    p_new_1 = half_bw_over_p + sqrt_disc
    p_new_2 = half_bw_over_p - sqrt_disc
    p_new = torch.cat([p_new_1, p_new_2])

    # Transform zeros (if any)
    if z.numel() > 0:
        half_bw_over_z = bw / (2 * z)
        disc_z = half_bw_over_z * half_bw_over_z - wo_sq
        sqrt_disc_z = torch.sqrt(disc_z.to(z.dtype))
        z_new_1 = half_bw_over_z + sqrt_disc_z
        z_new_2 = half_bw_over_z - sqrt_disc_z
        z_transformed = torch.cat([z_new_1, z_new_2])
    else:
        z_transformed = torch.empty(0, dtype=p.dtype, device=p.device)

    # Add zeros at ±j*wo for degree difference
    # Each original pole adds 2 zeros at ±j*wo
    j_wo = 1j * wo
    zeros_at_jwo = (
        torch.stack([j_wo, -j_wo]).expand(degree_diff, 2).reshape(-1)
    )
    zeros_at_jwo = zeros_at_jwo.to(p.dtype)
    z_new = torch.cat([z_transformed, zeros_at_jwo])

    # Adjust gain
    # For bandstop: k_new = k (no change in gain)
    k_new = k

    return z_new, p_new, k_new
