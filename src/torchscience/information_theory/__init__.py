"""Information theory operators."""

from ._jensen_shannon_divergence import jensen_shannon_divergence
from ._kullback_leibler_divergence import kullback_leibler_divergence
from ._shannon_entropy import shannon_entropy

__all__ = [
    "jensen_shannon_divergence",
    "kullback_leibler_divergence",
    "shannon_entropy",
]
