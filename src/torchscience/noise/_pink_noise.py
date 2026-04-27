import torch
from torch import Tensor

import torchscience._csrc # noqa: F401 - registers torch.ops.torchscience


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

    Generates pink noise with the given size.

    Pink noise is a type of noise that has a power spectral density that is inversely proportional to the frequency.
    This is in contrast to white noise, which has a power spectral density that is constant across all frequencies.
    Pink noise is often used in audio processing to simulate natural sounds like wind or water. 

    Algorithm
    ---------

    1. Generate white noise with the given size and generator.
    2. Compute the power spectral density of the white noise.
    3. Compute the pink noise by filtering the white noise with the power spectral density.
    4. Normalize the pink noise to have a power spectral density of 1.
    5. Return the pink noise.

    Testable Guarantees
    --------------------
    - The pink noise has a power spectral density that is inversely proportional to the frequency.

    Applications
    ------------
    - Audio processing
    
    Dtype Support
    -------------
    - Supports float16, bfloat16, float32, float64

    Parameters
    ----------
    size : int
        The size of the noise to generate.
    generator : Generator, optional
        The generator to use for the noise.
    out : Tensor, optional
        The output tensor to use for the noise. Not [yet?] supported.
    dtype : torch.dtype, optional
        The dtype of the noise. Must be a floating point dtype.
    layout : torch.layout, optional
        The layout of the noise. Must be strided.
    device : torch.device, optional
        The device of the noise. Default is CPU.
    requires_grad : bool, optional
        Whether the noise should be differentiable. Default is False.
    pin_memory : bool, optional
        Whether the noise should be pinned in memory. Default is False.
    memory_format : torch.memory_format, optional
        The memory format of the noise. Must be contiguous.

    Returns
    -------
    Tensor
        The generated pink noise.

    Examples
    --------
    Basic usage:

    >>> pink_noise(10)
    tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

    With a generator:

    >>> pink_noise(10, generator=torch.Generator(device="cpu").manual_seed(1))
    tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

    """

    if out is not None:
        raise NotImplementedError("pink_noise: out= is not supported")
    if layout != torch.strided:
        raise ValueError("pink_noise: only strided layout is supported")
    if memory_format != torch.contiguous_format:
        raise ValueError("pink_noise: only contiguous_format is supported")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    dt = dtype if dtype is not None else torch.get_default_dtype()
    anchor_kw = {}
    if pin_memory and dev.type == "cpu":
        anchor_kw["pin_memory"] = True
    anchor = torch.empty((), device=dev, dtype=dt, **anchor_kw)

    return torch.ops.torchscience.pink_noise(
        anchor,
        size,
        generator,
        dtype,
        layout,
        device,
        requires_grad,
        pin_memory,
    )
