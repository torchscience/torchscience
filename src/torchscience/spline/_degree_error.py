from torchscience.spline import SplineError


class DegreeError(SplineError):
    """Raised when degree is invalid for given knot count."""

    pass
