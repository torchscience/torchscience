"""Exceptions for the polynomial module."""


class PolynomialError(Exception):
    """Base exception for polynomial operations."""

    pass


class DegreeError(PolynomialError):
    """Raised when degree is invalid for operation."""

    pass
