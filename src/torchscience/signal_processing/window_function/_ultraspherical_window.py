from typing import Optional, Union

import torch
from torch import Tensor


def _gegenbauer_polynomial(n: int, mu: Tensor, x: Tensor) -> Tensor:
    """Evaluate Gegenbauer polynomial C_n^mu(x) using the recurrence relation.

    The Gegenbauer (ultraspherical) polynomials satisfy:
        C_0^mu(x) = 1
        C_1^mu(x) = 2*mu*x
        C_{k+1}^mu(x) = (2*(k+mu)/(k+1)) * x * C_k^mu(x)
                       - ((k+2*mu-1)/(k+1)) * C_{k-1}^mu(x)
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2.0 * mu * x

    c_prev = torch.ones_like(x)  # C_0
    c_curr = 2.0 * mu * x  # C_1

    for k in range(1, n):
        # Compute C_{k+1} from C_k and C_{k-1}
        a_k = 2.0 * (k + mu) / (k + 1.0)
        b_k = (k + 2.0 * mu - 1.0) / (k + 1.0)
        c_next = a_k * x * c_curr - b_k * c_prev
        c_prev = c_curr
        c_curr = c_next

    return c_curr


def ultraspherical_window(
    n: int,
    mu: Union[float, Tensor],
    x_mu: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Ultraspherical (Gegenbauer) window function (symmetric).

    Computes a symmetric ultraspherical window of length n. The ultraspherical
    window is a generalization of the Dolph-Chebyshev window that uses
    Gegenbauer (ultraspherical) polynomials, providing an additional degree
    of freedom to control the window shape.

    Mathematical Definition
    -----------------------
    The ultraspherical window is defined via its frequency response:

        W(theta) = C_{N-1}^{mu}(x_mu * cos(theta)) / C_{N-1}^{mu}(x_mu)

    where C_n^{mu}(x) is the Gegenbauer polynomial of degree n with parameter mu.
    The time-domain window is obtained by inverse FFT of this frequency response.

    Properties
    ----------
    - Generalizes the Dolph-Chebyshev window (mu -> 0 limit)
    - mu controls the trade-off between main lobe width and sidelobe decay
    - x_mu > 1 controls the sidelobe level
    - Larger mu: faster sidelobe decay, wider main lobe
    - Smaller mu: slower sidelobe decay (more uniform), narrower main lobe

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    mu : float or Tensor
        Gegenbauer polynomial parameter (mu > 0).
        Controls the sidelobe roll-off characteristics.
        - mu near 0: approaches Dolph-Chebyshev (equiripple sidelobes)
        - mu = 1: Saramaki window
        - mu = 2: faster sidelobe decay
    x_mu : float or Tensor
        Sidelobe level parameter (x_mu > 1).
        Controls the ratio of main lobe to sidelobe amplitude.
        Larger values give lower sidelobes but wider main lobe.
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
    This function supports autograd - gradients flow through mu and x_mu.

    The window coefficients are computed via:
    1. Evaluate the frequency response at N uniformly spaced points
    2. Apply inverse FFT to obtain time-domain coefficients
    3. Normalize so the center coefficient is 1

    References
    ----------
    .. [1] S. W. A. Bergen and A. Antoniou, "Design of Ultraspherical Window
           Functions with Prescribed Spectral Characteristics," EURASIP Journal
           on Applied Signal Processing, vol. 2004, no. 13, pp. 2053-2065, 2004.

    See Also
    --------
    periodic_ultraspherical_window : Periodic version for spectral analysis.
    dolph_chebyshev_window : Special case when mu -> 0.
    """
    if n < 0:
        raise ValueError(
            f"ultraspherical_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    # Convert parameters to tensors
    target_dtype = dtype or torch.float32

    if not isinstance(mu, Tensor):
        mu = torch.tensor(mu, dtype=target_dtype, device=device)
    if not isinstance(x_mu, Tensor):
        x_mu = torch.tensor(x_mu, dtype=target_dtype, device=device)

    # Ensure consistent dtype
    if mu.dtype != target_dtype:
        mu = mu.to(dtype=target_dtype)
    if x_mu.dtype != target_dtype:
        x_mu = x_mu.to(dtype=target_dtype)

    # Validate parameters
    if mu <= 0:
        raise ValueError(
            f"ultraspherical_window: mu must be positive, got {mu}"
        )
    if x_mu <= 1:
        raise ValueError(
            f"ultraspherical_window: x_mu must be > 1, got {x_mu}"
        )

    target_device = device or mu.device

    # For a symmetric N-point window, we use the Fourier series representation.
    # The window coefficient at position k (0 to N-1) is computed as a sum of
    # cosines weighted by the frequency response.
    #
    # For a symmetric window centered at (N-1)/2, we compute:
    # w[k] = sum_{m=0}^{M} c_m * cos(pi * m * (2*k - (N-1)) / N)
    #
    # where M = floor((N-1)/2) for odd N, or N/2 for even N,
    # and c_m = (2 - delta_{m,0} - delta_{m,N/2}) * W_m / N
    # with W_m being the frequency response.

    # Sample indices for window output
    k = torch.arange(n, dtype=target_dtype, device=target_device)

    # Compute window using direct Fourier series summation
    # For a symmetric window, we sum cosines centered at (N-1)/2
    center = (n - 1) / 2.0
    window = torch.zeros(n, dtype=target_dtype, device=target_device)

    # Number of frequency components to sum
    # For even N: use N/2 + 1 components (DC to Nyquist)
    # For odd N: use (N+1)/2 components
    n_freqs = n // 2 + 1

    for m in range(n_freqs):
        # Frequency (normalized angular frequency)
        omega = torch.pi * m / (n - 1) if n > 1 else torch.tensor(0.0)

        # Argument to Gegenbauer polynomial: x_mu * cos(omega)
        arg = x_mu * torch.cos(
            torch.tensor(omega, dtype=target_dtype, device=target_device)
        )

        # Evaluate frequency response: C_{N-1}^{mu}(arg) / C_{N-1}^{mu}(x_mu)
        c_n_arg = _gegenbauer_polynomial(n - 1, mu, arg.unsqueeze(0))
        c_n_x_mu = _gegenbauer_polynomial(n - 1, mu, x_mu.unsqueeze(0))
        freq_mag = c_n_arg / c_n_x_mu

        # Compute cosine term for each output sample
        # cos(omega * (k - center)) = cos(pi * m * (k - center) / (N-1))
        cosine_term = torch.cos(omega * (k - center))

        # Weighting: DC and Nyquist (if present) are weighted by 1, others by 2
        weight = 1.0 if (m == 0 or (n % 2 == 0 and m == n // 2)) else 2.0

        window = window + weight * freq_mag * cosine_term

    # Normalize so maximum is 1
    window = window / window.abs().max()

    if dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)

    return window
