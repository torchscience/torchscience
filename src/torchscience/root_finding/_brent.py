from typing import Callable

import torch
from torch import Tensor


def _get_default_tol(dtype: torch.dtype) -> float:
    """Get dtype-aware default tolerance."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    elif dtype == torch.float32:
        return 1e-6
    else:  # float64
        return 1e-12


class _BrentImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through root-finding."""

    @staticmethod
    def forward(ctx, root: Tensor, f_callable, orig_shape) -> Tensor:
        ctx.f_callable = f_callable
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None]:
        (root,) = ctx.saved_tensors

        # Compute df/dx at the root
        x = root.detach().requires_grad_(True)
        with torch.enable_grad():
            fx = ctx.f_callable(x)
            # Compute df/dx
            df_dx = torch.autograd.grad(
                fx,
                x,
                grad_outputs=torch.ones_like(fx),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute df/dtheta (gradient w.r.t. parameters)
            # This is done by computing the gradient of f w.r.t. its inputs
            # when evaluated at the root
            if fx.grad_fn is not None:
                # grad_output is dL/dx*, we need dL/dtheta
                # Using implicit function theorem: dx*/dtheta = -[df/dx]^{-1} * df/dtheta
                # So: dL/dtheta = dL/dx* * dx*/dtheta = -dL/dx* * [df/dx]^{-1} * df/dtheta
                #
                # We compute this by backpropagating through f with modified gradient
                modified_grad = -grad_output / df_dx
                torch.autograd.backward(fx, modified_grad)

        return None, None, None


def _attach_implicit_grad(result: Tensor, f, orig_shape: tuple) -> Tensor:
    """Attach implicit differentiation gradient if needed."""
    # Check if any parameter of f requires gradients
    try:
        test_input = result.detach().requires_grad_(True)
        with torch.enable_grad():
            test_output = f(test_input)
        needs_grad = test_output.requires_grad
    except Exception:
        needs_grad = False

    if not needs_grad:
        return result.reshape(orig_shape)

    # Make sure result has requires_grad=True for the autograd function
    # This is needed when result comes from endpoint detection (no iteration)
    if not result.requires_grad:
        result = result.clone().requires_grad_(True)

    result = _BrentImplicitGrad.apply(result, f, orig_shape)
    return result.reshape(orig_shape)


def brent(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> Tensor:
    """
    Find roots of f(x) = 0 using Brent's method.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape (N,), returns (N,).
    a, b : Tensor
        Bracket endpoints. Shape (N,). Must satisfy f(a) * f(b) < 0.
    xtol : float, optional
        Tolerance on interval width. Default: dtype-aware.
    ftol : float, optional
        Tolerance on |f(x)|. Default: dtype-aware.
    maxiter : int
        Maximum iterations. Raises RuntimeError if exceeded.

    Returns
    -------
    Tensor
        Roots of shape (N,).

    Raises
    ------
    ValueError
        If f(a) and f(b) have the same sign for any element.
    RuntimeError
        If convergence is not achieved within maxiter iterations.
    """
    # Input validation
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have same shape, got {a.shape} and {b.shape}"
        )

    # Flatten for processing, remember original shape
    orig_shape = a.shape

    if a.numel() == 0:
        return _attach_implicit_grad(a.clone(), f, orig_shape)

    a = a.flatten()
    b = b.flatten()

    # Get tolerances
    dtype = a.dtype
    if xtol is None:
        xtol = _get_default_tol(dtype)
    if ftol is None:
        ftol = _get_default_tol(dtype)

    # Evaluate function at endpoints
    fa = f(a)
    fb = f(b)

    # Check for NaN/Inf in inputs
    if torch.any(~torch.isfinite(a)) or torch.any(~torch.isfinite(b)):
        raise ValueError("a and b must not contain NaN or Inf")

    # Check for roots at endpoints first (before bracket validation)
    root = torch.where(fa == 0, a, torch.where(fb == 0, b, a.clone()))
    at_endpoint = (fa == 0) | (fb == 0)
    if torch.all(at_endpoint):
        return _attach_implicit_grad(root, f, orig_shape)

    # Check for valid brackets (only for non-endpoint cases)
    if torch.any(fa * fb >= 0):
        invalid = fa * fb >= 0
        raise ValueError(
            f"Invalid bracket: f(a) and f(b) must have opposite signs. "
            f"{invalid.sum().item()} of {invalid.numel()} brackets are invalid."
        )

    # Ensure |f(a)| >= |f(b)| by swapping if needed
    swap_mask = torch.abs(fa) < torch.abs(fb)
    a, b = torch.where(swap_mask, b, a), torch.where(swap_mask, a, b)
    fa, fb = torch.where(swap_mask, fb, fa), torch.where(swap_mask, fa, fb)

    # Initialize state
    c = a.clone()
    fc = fa.clone()
    d = torch.zeros_like(a)
    mflag = torch.ones(a.shape, dtype=torch.bool)

    # Track which elements have converged
    converged = at_endpoint.clone()
    result = root.clone()

    for iteration in range(maxiter):
        # Check convergence: both xtol AND ftol must be satisfied
        interval_small = torch.abs(b - a) < xtol
        residual_small = torch.abs(fb) < ftol
        newly_converged = interval_small & residual_small & ~converged
        converged = converged | newly_converged
        result = torch.where(newly_converged, b, result)

        if torch.all(converged):
            return _attach_implicit_grad(result, f, orig_shape)

        # Only update unconverged elements
        active = ~converged

        # Compute s using inverse quadratic interpolation or secant method
        # Inverse quadratic interpolation
        use_iqp = (fa != fc) & (fb != fc) & active
        s_iqp = torch.where(
            use_iqp,
            (a * fb * fc) / ((fa - fb) * (fa - fc))
            + (b * fa * fc) / ((fb - fa) * (fb - fc))
            + (c * fa * fb) / ((fc - fa) * (fc - fb)),
            torch.zeros_like(a),
        )

        # Secant method
        use_secant = ~use_iqp & active
        s_secant = torch.where(
            use_secant & (fb != fa),
            b - fb * (b - a) / (fb - fa),
            torch.zeros_like(a),
        )

        s = torch.where(use_iqp, s_iqp, s_secant)

        # Check if we should use bisection instead
        # Condition 1: s not in the acceptable range
        min_range = torch.minimum((3 * a + b) / 4, b)
        max_range = torch.maximum((3 * a + b) / 4, b)
        cond1 = ~((min_range < s) & (s < max_range))

        # Condition 2: mflag is true and |s - b| >= |b - c| / 2
        cond2 = mflag & (torch.abs(s - b) >= torch.abs(b - c) / 2)

        # Condition 3: mflag is false and |s - b| >= |c - d| / 2
        cond3 = ~mflag & (torch.abs(s - b) >= torch.abs(c - d) / 2)

        # Condition 4: mflag is true and |b - c| < xtol
        cond4 = mflag & (torch.abs(b - c) < xtol)

        # Condition 5: mflag is false and |c - d| < xtol
        cond5 = ~mflag & (torch.abs(c - d) < xtol)

        use_bisection = (cond1 | cond2 | cond3 | cond4 | cond5) & active

        # Bisection
        s = torch.where(use_bisection, (a + b) / 2, s)
        mflag = torch.where(active, use_bisection, mflag)

        # Evaluate function at s
        fs = f(s)

        # Check for NaN in function evaluation
        if torch.any(torch.isnan(fs) & active):
            raise RuntimeError("Function returned NaN during iteration")

        # d is now the value of c from the previous iteration
        d = torch.where(active, c, d)

        # c takes the value of b
        c = torch.where(active, b, c)
        fc = torch.where(active, fb, fc)

        # Update the bracket
        update_b = (fa * fs < 0) & active
        update_a = ~update_b & active

        # Update b
        b = torch.where(update_b, s, b)
        fb = torch.where(update_b, fs, fb)

        # Update a
        a = torch.where(update_a, s, a)
        fa = torch.where(update_a, fs, fa)

        # Ensure |f(a)| >= |f(b)| by swapping if needed
        swap = (torch.abs(fa) < torch.abs(fb)) & active
        a_new = torch.where(swap, b, a)
        b_new = torch.where(swap, a, b)
        fa_new = torch.where(swap, fb, fa)
        fb_new = torch.where(swap, fa, fb)

        a = a_new
        b = b_new
        fa = fa_new
        fb = fb_new

    raise RuntimeError(f"brent: failed to converge in {maxiter} iterations")
