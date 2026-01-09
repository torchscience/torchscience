from typing import Optional, Union

import torch
from torch import Tensor


def periodic_planck_taper_window(
    n: int,
    epsilon: Union[float, Tensor] = 0.1,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Planck-taper window function (periodic).

    Computes a periodic Planck-taper window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic Planck-taper window is defined for normalized position
    t = k / n where k = 0, 1, ..., n-1:

    For 0 < t < epsilon:
        w(t) = 1 / (1 + exp(Z⁺(t)))
        where Z⁺(t) = epsilon * (1/t + 1/(t - epsilon))

    For epsilon <= t <= 1 - epsilon:
        w(t) = 1

    For 1 - epsilon < t < 1:
        w(t) = 1 / (1 + exp(Z⁻(t)))
        where Z⁻(t) = epsilon * (1/(1 - t) + 1/(1 - t - epsilon))

    At the boundary t = 0:
        w(t) = 0

    Properties
    ----------
    - Infinitely differentiable (C∞) at all points
    - epsilon controls the fraction of the window that is tapered on each side
    - epsilon = 0: rectangular window (all ones)
    - epsilon = 0.5: fully tapered (no flat region)
    - Designed for spectral analysis with FFT

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    epsilon : float or Tensor, optional
        Taper parameter controlling the fraction of the window inside the
        tapered region on each side. Must be in [0, 0.5]. Default is 0.1.
        - epsilon = 0: rectangular window
        - epsilon = 0.5: fully tapered window
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    Notes
    -----
    This function supports autograd - gradients flow through the epsilon
    parameter.

    See Also
    --------
    planck_taper_window : Symmetric version.
    periodic_tukey_window : Periodic cosine-tapered window.
    """
    if n < 0:
        raise ValueError(
            f"periodic_planck_taper_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(epsilon, Tensor):
        target_dtype = dtype or torch.float32
        epsilon = torch.tensor(epsilon, dtype=target_dtype, device=device)

    # For periodic window, denominator is n
    denom = float(n)
    k = torch.arange(n, dtype=epsilon.dtype, device=epsilon.device)
    t = k / denom

    # Initialize window to ones (flat region)
    window = torch.ones_like(t)

    # Handle epsilon = 0 case (rectangular window)
    if epsilon == 0:
        if dtype is not None and window.dtype != dtype:
            window = window.to(dtype=dtype)
        return window

    # Left taper region: 0 < t < epsilon
    # Z⁺(t) = epsilon * (1/t + 1/(t - epsilon))
    left_mask = (t > 0) & (t < epsilon)
    t_left = t[left_mask]
    z_left = epsilon * (1.0 / t_left + 1.0 / (t_left - epsilon))
    window[left_mask] = 1.0 / (1.0 + torch.exp(z_left))

    # Right taper region: 1 - epsilon < t < 1
    # Z⁻(t) = epsilon * (1/(1-t) + 1/(1-t-epsilon))
    right_mask = (t > 1 - epsilon) & (t < 1)
    t_right = t[right_mask]
    z_right = epsilon * (
        1.0 / (1.0 - t_right) + 1.0 / (1.0 - t_right - epsilon)
    )
    window[right_mask] = 1.0 / (1.0 + torch.exp(z_right))

    # Boundary point: t = 0 is exactly 0
    window[0] = 0.0

    if dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)

    return window
