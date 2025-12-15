import warnings

import torch
from torch import Tensor


def chebyshev_polynomial_t(v: Tensor, z: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    Chebyshev polynomial of the first kind.

    Computes the Chebyshev polynomial of the first kind T_v(z).

    Mathematical Definition
    -----------------------
    For integer v = n >= 0:
        Uses the recurrence relation (exact polynomial semantics):
            T_0(z) = 1
            T_1(z) = z
            T_n(z) = 2z * T_{n-1}(z) - T_{n-2}(z)

    For non-integer v (real or complex), or complex z:
        Uses the analytic continuation:
            T_v(z) = cos(v * arccos(z))

        where arccos(z) uses the principal branch, consistent with PyTorch's
        complex acos.

    Branch Conventions
    ------------------
    - Uses the principal branch of arccos(z), consistent with torch.acos.
    - For real z in [-1, 1]: arccos(z) is real in [0, pi].
    - For real z outside [-1, 1]: arccos(z) is complex.
    - For complex z: arccos(z) follows the principal branch with branch cuts
      at (-inf, -1) and (1, +inf) on the real axis.

    Dispatch Logic
    --------------
    - Path A (Recurrence): v is integral AND z is real -> polynomial recurrence.
      Numerically stable for integer degrees, exact polynomial semantics.
    - Path B (Analytic Continuation): non-integer v OR complex v OR complex z
      -> uses cos(v * acos(z)) with principal branch.

    Dtype Promotion
    ---------------
    - If either v or z is complex -> output is complex.
    - complex64 if all inputs <= float32/complex64, else complex128.
    - If both real -> standard PyTorch promotion rules apply.
    - Supports float16, bfloat16, float32, float64, complex64, complex128.

    Integer Dtype Handling
    ----------------------
    If v is passed as an integer dtype tensor (e.g., torch.int32, torch.int64),
    it will be promoted to a floating-point dtype via PyTorch's standard type
    promotion rules before computation. The promoted dtype is determined by
    the dtype of z.

    To use the efficient recurrence path for integer degrees, pass v as a
    floating-point tensor with integer values (e.g., torch.tensor([2.0]) rather
    than torch.tensor([2])). The implementation detects integer values by
    checking if v == floor(v) and uses the recurrence relation for such values,
    regardless of the underlying dtype.

    Examples of dtype handling:
        - v=torch.tensor([2], dtype=torch.int64), z=torch.tensor([0.5]) ->
          v promoted to float32, uses recurrence path (v has integer value)
        - v=torch.tensor([2.0]), z=torch.tensor([0.5]) ->
          stays float32, uses recurrence path
        - v=torch.tensor([2.5]), z=torch.tensor([0.5]) ->
          uses analytic continuation (non-integer v)

    Autograd Support
    ----------------
    - Gradients for z are always computed when z.requires_grad is True.
    - Gradients for v are only computed when v is floating-point or complex
      and v.requires_grad is True. No gradients are provided for integral v.
    - Second-order derivatives (gradgradcheck) are fully supported, enabling
      use in applications requiring Hessians or higher-order optimization.

    Backward formulas:
        dT_v(z)/dz = v * sin(v * arccos(z)) / sqrt(1 - z^2)
        dT_v(z)/dv = -sin(v * arccos(z)) * arccos(z)

    Parameters
    ----------
    v : Tensor
        Degree of the polynomial. Can be integral, floating-point, or complex.
        When integral, uses efficient polynomial recurrence for real z.
        When floating or complex, uses analytic continuation.
    z : Tensor
        Input tensor. Can be floating-point or complex.
        Broadcasting with v is supported.
    out : Tensor, optional
        Output tensor to write the result to.

    Returns
    -------
    Tensor
        The Chebyshev polynomial T_v(z) evaluated at the input values.
        Output dtype follows the promotion rules described above.

    Examples
    --------
    Integer degree with real input (recurrence path):

    >>> v = torch.tensor([0, 1, 2, 3])
    >>> z = torch.tensor([0.5])
    >>> chebyshev_polynomial_t(v, z)
    tensor([ 1.0000,  0.5000, -0.5000, -1.0000])

    Non-integer degree (analytic continuation):

    >>> v = torch.tensor([0.5, 1.5, 2.5])
    >>> z = torch.tensor([0.0])
    >>> chebyshev_polynomial_t(v, z)  # cos(v * pi/2)
    tensor([ 0.7071,  0.0000, -0.7071])

    Complex input:

    >>> v = torch.tensor([2.0])
    >>> z = torch.tensor([1.0 + 0.1j])
    >>> chebyshev_polynomial_t(v, z)  # Returns complex result
    tensor([1.0200-0.4000j])

    Autograd with floating v:

    >>> v = torch.tensor([2.0], requires_grad=True)
    >>> z = torch.tensor([0.5], requires_grad=True)
    >>> y = chebyshev_polynomial_t(v, z)
    >>> y.backward()
    >>> v.grad  # Gradient w.r.t. degree
    tensor([-0.9069])
    >>> z.grad  # Gradient w.r.t. input
    tensor([-2.0000])
    """
    output = torch.ops.torchscience.chebyshev_polynomial_t(v, z)

    if out is not None:
        out.resize_as_(output)

        out.copy_(output)

        return out

    return output
