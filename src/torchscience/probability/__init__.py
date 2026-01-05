"""Probability distributions with differentiable CDF, PDF, PPF, and SF.

This module provides functional operators for probability distributions,
complementing torch.distributions with:
- CDFs for all distributions (not just 6)
- PPF (quantile/inverse CDF) functions
- Second-order gradient support for all operators

Example
-------
>>> import torch
>>> from torchscience.probability import normal_cdf, normal_ppf
>>>
>>> # Compute CDF
>>> x = torch.tensor([0.0, 1.0, 2.0])
>>> p = normal_cdf(x)  # tensor([0.5, 0.8413, 0.9772])
>>>
>>> # Compute quantiles
>>> probs = torch.tensor([0.025, 0.5, 0.975])
>>> quantiles = normal_ppf(probs)  # tensor([-1.96, 0.0, 1.96])
"""

from ._exceptions import DomainError, ProbabilityError
from ._normal import (
    normal_cdf,
    normal_logpdf,
    normal_pdf,
    normal_ppf,
    normal_sf,
)

__all__ = [
    "DomainError",
    "ProbabilityError",
    "normal_cdf",
    "normal_logpdf",
    "normal_pdf",
    "normal_ppf",
    "normal_sf",
]
