from typing import Optional, Union

import torch
from torch import Tensor


def periodic_exponential_window(
    n: int,
    tau: Union[float, Tensor] = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Exponential (Poisson) window function (periodic).

    Computes a periodic exponential window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic exponential window is defined as:

        w[k] = exp(-|k - center| / tau)

    for k = 0, 1, ..., n-1, where center = n/2 for the periodic version.

    Properties
    ----------
    - Exponential decay from center
    - tau controls the decay rate (larger tau = slower decay)
    - Designed for spectral analysis with FFT

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    tau : float or Tensor, optional
        Decay parameter controlling the window shape. Default is 1.0.
        Larger values produce slower decay (wider window).
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
    This function supports autograd - gradients flow through the tau parameter.

    See Also
    --------
    exponential_window : Symmetric version.
    """
    if n < 0:
        raise ValueError(
            f"periodic_exponential_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(tau, Tensor):
        target_dtype = dtype or torch.float32
        tau = torch.tensor(tau, dtype=target_dtype, device=device)

    # For periodic window, center = n / 2
    center = n / 2.0

    k = torch.arange(n, dtype=tau.dtype, device=tau.device)

    # w[k] = exp(-|k - center| / tau)
    window = torch.exp(-torch.abs(k - center) / tau)

    if dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)

    return window
