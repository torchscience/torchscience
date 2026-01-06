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

from ._beta import (
    beta_cdf,
    beta_pdf,
    beta_ppf,
)
from ._binomial import (
    binomial_cdf,
    binomial_pmf,
)
from ._chi2 import (
    chi2_cdf,
    chi2_pdf,
    chi2_ppf,
    chi2_sf,
)
from ._exceptions import DomainError, ProbabilityError
from ._f import (
    f_cdf,
    f_pdf,
    f_ppf,
    f_sf,
)
from ._gamma import (
    gamma_cdf,
    gamma_pdf,
    gamma_ppf,
)
from ._normal import (
    normal_cdf,
    normal_logpdf,
    normal_pdf,
    normal_ppf,
    normal_sf,
)
from ._poisson import (
    poisson_cdf,
    poisson_pmf,
)

__all__ = [
    "DomainError",
    "ProbabilityError",
    # Beta distribution
    "beta_cdf",
    "beta_pdf",
    "beta_ppf",
    # Chi-squared distribution
    "chi2_cdf",
    "chi2_pdf",
    "chi2_ppf",
    "chi2_sf",
    # F distribution
    "f_cdf",
    "f_pdf",
    "f_ppf",
    "f_sf",
    # Gamma distribution
    "gamma_cdf",
    "gamma_pdf",
    "gamma_ppf",
    # Normal distribution
    "normal_cdf",
    "normal_logpdf",
    "normal_pdf",
    "normal_ppf",
    "normal_sf",
    # Binomial distribution
    "binomial_cdf",
    "binomial_pmf",
    # Poisson distribution
    "poisson_cdf",
    "poisson_pmf",
]
