import torch
from torch import Tensor

from ._common import dispatch_colored_noise


def blue_noise(
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
    Blue noise.

    Generates blue noise (also called azure noise) of the given size. Blue
    noise has a power spectral density proportional to frequency:

    .. math::
        S(f) \propto f.

    It is the spectral mirror of pink noise: high frequencies are
    emphasized. Power per octave quadruples for each octave shift.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Multiply each bin by ``sqrt(|f|)``, giving |X(f)| ~ sqrt(f) and hence
       |X(f)|^2 ~ f.
    4. The DC bin is naturally zero because ``sqrt(0) = 0``; the time-domain
       output is therefore exactly zero-mean by construction.
    5. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - PSD log-log slope equals +1 (slope test).
    - Power quadruples per octave (octave-power-ratio test).
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Dithering and stippling: perceptually preferable to white noise
      because the high-frequency emphasis is less visible / audible to
      humans.
    - Halftone-style image dithering and Monte-Carlo sampling patterns.

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
        The generated blue noise, shape ``(size,)``.

    Examples
    --------
    >>> blue_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    return dispatch_colored_noise(
        "blue_noise",
        torch.ops.torchscience.blue_noise,
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
