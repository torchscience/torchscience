import torch
from torch import Tensor

from ._common import dispatch_colored_noise


def white_noise(
    size: int,
    *,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
) -> Tensor:
    r"""
    White noise.

    Generates white noise of the given size. White noise has a flat power
    spectral density:

    .. math::
        S(f) = \mathrm{const}.

    Each frequency contributes equal power, so the total power in a band
    ``[f_0, 2 f_0]`` doubles for every octave (energy per octave grows as
    ``f``). Time-domain samples are independent draws from a normal
    distribution, then made exactly zero-mean and rescaled to unit peak
    amplitude.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Zero the DC bin -- this makes the time-domain output exactly
       zero-mean. (The white spectrum on every non-DC bin is preserved.)
    4. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - PSD log-log slope is 0 (flat spectrum).
    - Power doubles for every octave shift in frequency.
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Baseline noise for system identification and dithering.
    - Source noise for filter banks generating other colored-noise signals.

    Dtype Support
    -------------
    - Supports float16, bfloat16, float32, float64.

    Parameters
    ----------
    size : int
        Length of the output 1-D tensor.
    generator : Generator, optional
        Random generator. Same seed -> same output.
    out : Tensor, optional
        Not supported; ``out=`` raises ``NotImplementedError``.
    dtype : torch.dtype, optional
        Floating-point dtype. Default: torch default dtype.
    layout : torch.layout, optional
        Must be ``torch.strided``.
    device : torch.device, optional
        Default is CPU.
    requires_grad : bool, optional
        Default is False.
    pin_memory : bool, optional
        Default is False.
    memory_format : torch.memory_format, optional
        Must be ``torch.contiguous_format``.

    Returns
    -------
    Tensor
        The generated white noise, shape ``(size,)``.

    Examples
    --------
    >>> white_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    return dispatch_colored_noise(
        "white_noise",
        torch.ops.torchscience.white_noise,
        size,
        generator=generator,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
