"""Bilinear transform for analog to digital filter conversion."""

from typing import Tuple, Union

import torch
from torch import Tensor


def bilinear_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    fs: Union[float, Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Transform an analog filter to a digital filter using bilinear transform.

    The bilinear transform maps the s-plane to the z-plane using:
    s = (2*fs) * (z - 1) / (z + 1)

    Parameters
    ----------
    z : Tensor
        Zeros of the analog filter.
    p : Tensor
        Poles of the analog filter.
    k : Tensor
        System gain of the analog filter.
    fs : float or Tensor
        Sampling frequency (Hz).

    Returns
    -------
    z_d : Tensor
        Zeros of the digital filter.
    p_d : Tensor
        Poles of the digital filter.
    k_d : Tensor
        System gain of the digital filter.

    Notes
    -----
    The bilinear transform:
    - Maps left half-plane (stable analog) to inside unit circle (stable digital)
    - Maps imaginary axis to unit circle
    - Introduces frequency warping: omega_d = 2*fs * arctan(omega_a / (2*fs))
    - Adds zeros at z=-1 for each degree difference (all-pole analog -> FIR zeros)

    For frequency prewarping (not included here), prewarp the analog
    filter before calling bilinear_zpk.
    """
    if not isinstance(fs, Tensor):
        fs = torch.as_tensor(fs, dtype=k.dtype, device=k.device)

    fs2 = 2 * fs  # 2 * sampling frequency

    degree_diff = p.numel() - z.numel()

    # Transform poles: z = (1 + s/(2*fs)) / (1 - s/(2*fs))
    p_d = (1 + p / fs2) / (1 - p / fs2)

    # Transform existing zeros
    if z.numel() > 0:
        z_d_transformed = (1 + z / fs2) / (1 - z / fs2)
    else:
        z_d_transformed = torch.empty(0, dtype=p.dtype, device=p.device)

    # Add zeros at z=-1 (Nyquist) for degree difference
    zeros_at_nyquist = -torch.ones(degree_diff, dtype=p.dtype, device=p.device)
    z_d = torch.cat([z_d_transformed, zeros_at_nyquist])

    # Adjust gain
    # k_d = k * real(prod(fs2 - z) / prod(fs2 - p))
    if z.numel() > 0:
        num = torch.prod(fs2 - z)
    else:
        num = torch.tensor(1.0, dtype=p.dtype, device=p.device)
    den = torch.prod(fs2 - p)
    k_d = k * torch.real(num / den)

    return z_d, p_d, k_d
