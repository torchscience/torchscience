"""Exceptions for the spline module."""


class SplineError(Exception):
    """Base exception for spline operations."""

    pass


class ExtrapolationError(SplineError):
    """Raised when query point is outside spline domain with extrapolate='error'."""

    pass


class KnotError(SplineError):
    """Raised for invalid knot vectors (non-monotonic, insufficient knots)."""

    pass


class DegreeError(SplineError):
    """Raised when degree is invalid for given knot count."""

    pass
