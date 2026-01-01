"""Exceptions for ODE solvers."""


class ODESolverError(Exception):
    """Base exception for ODE solver errors."""

    pass


class MaxStepsExceeded(ODESolverError):
    """Raised when adaptive solver exceeds max_steps."""

    pass


class StepSizeError(ODESolverError):
    """Raised when adaptive step size falls below dt_min."""

    pass


class ConvergenceError(ODESolverError):
    """Raised when implicit solver Newton iteration fails to converge."""

    pass
