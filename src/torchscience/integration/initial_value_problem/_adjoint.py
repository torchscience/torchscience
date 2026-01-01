"""Adjoint method wrapper for memory-efficient ODE gradients."""

from typing import Any, Callable, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from torchscience.integration.initial_value_problem._tensordict_utils import (
    flatten_state,
)


class _AdjointODEFunction(torch.autograd.Function):
    """
    Custom autograd function implementing the continuous adjoint method.

    Forward pass: Solve the ODE normally
    Backward pass: Solve augmented adjoint ODE backwards in time
    """

    @staticmethod
    def forward(
        ctx,
        y0_flat: torch.Tensor,
        t0: float,
        t1: float,
        solver: Callable,
        f_flat: Callable,
        solver_kwargs: dict,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve ODE and store info for backward.

        Parameters
        ----------
        ctx : Context
            Autograd context for saving tensors
        y0_flat : Tensor
            Flattened initial state
        t0, t1 : float
            Integration interval
        solver : callable
            ODE solver function
        f_flat : callable
            Dynamics function for flattened state
        solver_kwargs : dict
            Additional kwargs for solver
        *params : Tensors
            Learnable parameters to compute gradients for

        Returns
        -------
        y_final_flat : Tensor
            Flattened final state
        """
        with torch.no_grad():
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **solver_kwargs
            )

        # Save for backward
        ctx.save_for_backward(y0_flat, y_final, *params)
        ctx.solver = solver
        ctx.f_flat = f_flat
        ctx.solver_kwargs = solver_kwargs
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.interp = interp
        ctx.n_params = len(params)

        return y_final.clone()

    @staticmethod
    def backward(ctx, grad_y_final: torch.Tensor) -> Tuple[Any, ...]:
        """
        Backward pass: solve adjoint ODE to compute gradients.

        The adjoint equation is:
            da/dt = -a^T @ (df/dy)

        where a is the adjoint variable (gradient w.r.t. state).

        Parameter gradients are computed as:
            dL/dtheta = integral_t0^t1 a^T @ (df/dtheta) dt
        """
        saved = ctx.saved_tensors
        y0_flat = saved[0]
        y_final = saved[1]
        params = saved[2:]

        solver = ctx.solver
        f_flat = ctx.f_flat
        solver_kwargs = ctx.solver_kwargs
        t0, t1 = ctx.t0, ctx.t1
        interp = ctx.interp
        n_params = ctx.n_params

        # Initial adjoint state (gradient w.r.t. final state)
        a = grad_y_final.clone()

        # Accumulate parameter gradients
        param_grads = [torch.zeros_like(p) for p in params]

        # Solve adjoint ODE backwards
        # We approximate by using the stored interpolant and stepping backwards

        # Number of steps for adjoint integration
        n_steps = 100
        dt = (t1 - t0) / n_steps

        t = t1

        for _ in range(n_steps):
            t_prev = t - dt

            # Get state at this time from interpolant
            with torch.enable_grad():
                y_at_t = interp(t)
                if not y_at_t.requires_grad:
                    y_at_t = y_at_t.clone().requires_grad_(True)

                # Compute f and its Jacobians
                f_val = f_flat(t, y_at_t)

            # Adjoint update: da/dt = -a^T @ (df/dy)
            # Approximate: a_prev = a + dt * a^T @ (df/dy)
            # Use vector-Jacobian product for efficiency
            if y_at_t.requires_grad and f_val.requires_grad:
                # Compute VJP: a^T @ (df/dy)
                vjp_result = torch.autograd.grad(
                    f_val,
                    y_at_t,
                    grad_outputs=a,
                    retain_graph=True,
                    allow_unused=True,
                )
                if vjp_result[0] is not None:
                    a = a + dt * vjp_result[0]

            # Accumulate parameter gradients
            for i, p in enumerate(params):
                if p.requires_grad and f_val.requires_grad:
                    # Compute a^T @ (df/dtheta)
                    grads = torch.autograd.grad(
                        f_val,
                        p,
                        grad_outputs=a,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    if grads[0] is not None:
                        param_grads[i] = param_grads[i] + dt * grads[0]

            t = t_prev

        # Gradient w.r.t. y0 is the final adjoint state
        grad_y0 = a

        # Return gradients in same order as forward inputs
        # (y0_flat, t0, t1, solver, f_flat, solver_kwargs, *params)
        return (grad_y0, None, None, None, None, None) + tuple(param_grads)


def adjoint(
    solver: Callable,
    checkpoints: Optional[int] = None,
) -> Callable:
    """
    Wrap a solver to use adjoint method for gradients.

    The adjoint method computes gradients by solving an augmented ODE
    backwards in time, using O(1) memory for the autograd graph instead
    of O(n_steps).

    Parameters
    ----------
    solver : callable
        Any ODE solver function (euler, dormand_prince_5, etc.)
    checkpoints : int, optional
        Number of checkpoints for memory/compute tradeoff.
        None = automatic selection (currently unused, reserved for future).

    Returns
    -------
    wrapped_solver : callable
        Solver with same signature but using adjoint gradients.

    Examples
    --------
    >>> from torchscience.integration.initial_value_problem import (
    ...     adjoint,
    ...     dormand_prince_5,
    ... )
    >>> adjoint_solver = adjoint(dormand_prince_5)
    >>> y_final, interp = adjoint_solver(f, y0, t_span=(0.0, 10.0))
    >>> loss = y_final.sum()
    >>> loss.backward()  # Uses O(1) memory for autograd graph

    Notes
    -----
    The adjoint method only affects gradient computation. It does NOT change:

    - The interpolant (still stores trajectory points for dense output)
    - Forward solve behavior (same numerical solution)
    - Return values (same (y_final, interp) tuple)

    Memory savings come from not storing the autograd computation graph,
    not from discarding the trajectory.

    When to use:

    - Long integrations with large state dimension
    - Memory-constrained environments
    - When exact discretization gradients are not required

    When NOT to use:

    - Short integrations (overhead not worth it)
    - When you need exact discretization gradients
    - When differentiating through the interpolant
    """

    def wrapped_solver(
        f: Callable,
        y0: Union[torch.Tensor, TensorDict],
        t_span: Tuple[float, float],
        **kwargs,
    ) -> Tuple[
        Union[torch.Tensor, TensorDict],
        Callable[
            [Union[float, torch.Tensor]], Union[torch.Tensor, TensorDict]
        ],
    ]:
        """Wrapped solver using adjoint method for gradients."""
        t0, t1 = t_span

        # Handle TensorDict
        is_tensordict = isinstance(y0, TensorDict)
        y0_flat, unflatten = flatten_state(y0)

        if is_tensordict:

            def f_flat(t, y):
                y_struct = unflatten(y)
                dy_struct = f(t, y_struct)
                dy_flat, _ = flatten_state(dy_struct)
                return dy_flat

        else:
            f_flat = f

        # Find all parameters that require gradients
        # This is tricky - we need to capture them from f's closure
        # For now, we rely on the dynamics function using module parameters

        # Get parameters from dynamics function closure
        params = []

        def extract_params(obj, seen=None):
            """Recursively extract tensors requiring grad from closures."""
            if seen is None:
                seen = set()
            if id(obj) in seen:
                return
            seen.add(id(obj))

            if isinstance(obj, torch.Tensor) and obj.requires_grad:
                params.append(obj)
            elif hasattr(obj, "__closure__") and obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        extract_params(cell.cell_contents, seen)
                    except ValueError:
                        pass
            elif hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    extract_params(v, seen)

        extract_params(f)

        # If no parameters found, just do regular forward (no gradients needed)
        if not params:
            y_final, interp = solver(
                f_flat, y0_flat, t_span=(t0, t1), **kwargs
            )
            if is_tensordict:
                y_final = unflatten(y_final)

                def interp_td(t_query):
                    y_flat_query = interp(t_query)
                    if isinstance(t_query, (int, float)) or (
                        isinstance(t_query, torch.Tensor)
                        and t_query.dim() == 0
                    ):
                        return unflatten(y_flat_query)
                    return torch.stack(
                        [
                            unflatten(y_flat_query[i])
                            for i in range(y_flat_query.shape[0])
                        ]
                    )

                interp_td.success = getattr(interp, "success", None)
                return y_final, interp_td
            return y_final, interp

        # Use custom autograd function for adjoint gradients
        # Pass params as *args so they're tracked by autograd
        y_final_flat = _AdjointODEFunction.apply(
            y0_flat,
            t0,
            t1,
            solver,
            f_flat,
            kwargs,
            *params,
        )

        # Get interpolant from a separate forward pass (with no_grad)
        with torch.no_grad():
            _, interp = solver(f_flat, y0_flat, t_span=(t0, t1), **kwargs)

        # Unflatten result
        if is_tensordict:
            y_final = unflatten(y_final_flat)

            def interp_tensordict(t_query):
                y_flat_query = interp(t_query)
                if isinstance(t_query, (int, float)) or (
                    isinstance(t_query, torch.Tensor) and t_query.dim() == 0
                ):
                    return unflatten(y_flat_query)
                return torch.stack(
                    [
                        unflatten(y_flat_query[i])
                        for i in range(y_flat_query.shape[0])
                    ]
                )

            interp_tensordict.success = getattr(interp, "success", None)
            return y_final, interp_tensordict

        return y_final_flat, interp

    # Preserve solver metadata
    wrapped_solver.__name__ = (
        f"adjoint({getattr(solver, '__name__', 'solver')})"
    )
    wrapped_solver.__doc__ = f"""
    {getattr(solver, "__name__", "Solver")} with adjoint method for memory-efficient gradients.

    This is a wrapped version of {getattr(solver, "__name__", "the solver")} that uses
    the continuous adjoint method to compute gradients with O(1) memory instead of
    O(n_steps).

    See `adjoint()` documentation for details.
    """

    return wrapped_solver
