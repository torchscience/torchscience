from torchscience.polynomial import PolynomialError


class DegreeError(PolynomialError):
    """Raised when degree is invalid for operation."""

    pass
