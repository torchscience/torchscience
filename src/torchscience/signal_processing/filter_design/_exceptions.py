"""Exceptions for filter design module."""


class FilterDesignError(Exception):
    """Base exception for filter design errors."""

    pass


class InvalidFilterOrderError(FilterDesignError):
    """Raised when filter order is invalid."""

    pass


class InvalidCutoffError(FilterDesignError):
    """Raised when cutoff frequency is invalid."""

    pass
