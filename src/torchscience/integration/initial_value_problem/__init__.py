"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Available Solvers
-----------------
euler
    Forward Euler method (1st order, fixed step, explicit).
    Simplest method, educational baseline.

midpoint
    Explicit midpoint method (2nd order, fixed step).
    Good accuracy/cost tradeoff for smooth problems.

runge_kutta_4
    Classic 4th-order Runge-Kutta (fixed step, explicit).
    Widely used workhorse, excellent for non-stiff problems.

dormand_prince_5
    Dormand-Prince 5(4) adaptive method (explicit).
    Production-quality solver with error control.

backward_euler
    Backward Euler method (1st order, fixed step, implicit).
    A-stable, suitable for stiff problems.

Examples
--------
>>> import torch
>>> from torchscience.integration.initial_value_problem import backward_euler
>>>
>>> def stiff_decay(t, y):
...     return -1000 * y  # Stiff coefficient
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = backward_euler(stiff_decay, y0, t_span=(0.0, 1.0), dt=0.1)
"""

from torchscience.integration.initial_value_problem._backward_euler import (
    backward_euler,
)
from torchscience.integration.initial_value_problem._dormand_prince_5 import (
    dormand_prince_5,
)
from torchscience.integration.initial_value_problem._euler import euler
from torchscience.integration.initial_value_problem._exceptions import (
    ConvergenceError,
    MaxStepsExceeded,
    ODESolverError,
    StepSizeError,
)
from torchscience.integration.initial_value_problem._midpoint import midpoint
from torchscience.integration.initial_value_problem._runge_kutta_4 import (
    runge_kutta_4,
)

__all__ = [
    # Exceptions
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    # Explicit solvers (ordered by complexity)
    "euler",
    "midpoint",
    "runge_kutta_4",
    "dormand_prince_5",
    # Implicit solvers
    "backward_euler",
]
