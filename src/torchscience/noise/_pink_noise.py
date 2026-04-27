import torch
from torch import Tensor

from ._common import dispatch_colored_noise


def pink_noise(
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
    Pink noise.

    Generates pink noise of the given size. Pink noise has a power spectral
    density that is inversely proportional to frequency:

    .. math::
        S(f) \propto \frac{1}{f}.

    A direct consequence is that every octave band carries (approximately) the
    same total power -- the "equal energy per octave" property. Pink noise is
    common in audio (it sounds natural, like wind or rain), in 1/f flicker
    noise found in many physical systems, and in pinkening of integrators or
    flicker-driven processes.

    Algorithm
    ---------

    1. Draw N i.i.d. samples from N(0, 1).
    2. Take the real FFT.
    3. Divide each non-DC bin by sqrt(|f|), giving |X(f)| ~ 1/sqrt(f) and
       hence |X(f)|^2 ~ 1/f.
    4. Set the DC bin (f=0) to zero -- 1/f is undefined at f=0, and zeroing
       DC also makes the time-domain output exactly zero-mean.
    5. Inverse FFT, then normalize by max absolute value so peak amplitude
       is 1.

    Testable Guarantees
    -------------------
    - PSD log-log slope equals -1 (slope test).
    - Equal total power per octave (octave-power-ratio test).
    - Mean is zero up to FFT round-off.
    - Peak absolute value is exactly 1.

    Applications
    ------------
    - Audio mastering and synthesis (natural-sounding noise).
    - Modeling 1/f flicker noise.

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
        The generated pink noise, shape ``(size,)``.

    Examples
    --------
    >>> pink_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    """
    return dispatch_colored_noise(
        "pink_noise",
        torch.ops.torchscience.pink_noise,
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
