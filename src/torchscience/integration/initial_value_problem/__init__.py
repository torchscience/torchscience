"""
Initial value problem solvers for ordinary differential equations.

This module provides differentiable ODE solvers for PyTorch tensors and TensorDict.

Available Solvers
-----------------
dormand_prince_5
    Adaptive 5th-order Runge-Kutta method (Dormand-Prince 5(4)).
    Production-quality solver for most non-stiff problems.

Examples
--------
>>> import torch
>>> from torchscience.integration.initial_value_problem import dormand_prince_5
>>>
>>> def decay(t, y):
...     return -y
>>>
>>> y0 = torch.tensor([1.0])
>>> y_final, interp = dormand_prince_5(decay, y0, t_span=(0.0, 5.0))
>>> trajectory = interp(torch.linspace(0, 5, 100))
"""

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

__all__ = [
    "ConvergenceError",
    "MaxStepsExceeded",
    "ODESolverError",
    "StepSizeError",
    "dormand_prince_5",
    "euler",
]
