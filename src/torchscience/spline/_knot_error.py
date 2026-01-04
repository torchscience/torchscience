from torchscience.spline import SplineError


class KnotError(SplineError):
    """Raised for invalid knot vectors (non-monotonic, insufficient knots)."""

    pass
