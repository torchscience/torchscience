import torch


def hypergeometric_2_f_1(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian hypergeometric function ₂F₁(a, b; c; z).

    The Gaussian hypergeometric function is defined as:

        ₂F₁(a, b; c; z) = Σ(n=0 to ∞) [(a)ₙ(b)ₙ / (c)ₙ] * (zⁿ / n!)

    where (x)ₙ is the Pochhammer symbol (rising factorial):
        (x)ₙ = x(x+1)(x+2)...(x+n-1)

    Args:
        a: First numerator parameter (Tensor)
        b: Second numerator parameter (Tensor)
        c: Denominator parameter (Tensor), must not be a non-positive integer
        z: Argument (Tensor), typically |z| < 1 for convergence

    Returns:
        Tensor: The value of ₂F₁(a, b; c; z)

    Example:
        >>> import torch
        >>> from torchscience.special_functions import hypergeometric_2_f_1
        >>> a = torch.tensor([1.0])
        >>> b = torch.tensor([2.0])
        >>> c = torch.tensor([3.0])
        >>> z = torch.tensor([0.5])
        >>> result = hypergeometric_2_f_1(a, b, c, z)

    Note:
        - The function supports autograd for computing gradients
        - All inputs should be tensors with the same dtype and device
        - The series converges for |z| < 1 and has special behavior at |z| = 1
    """
    return torch.ops.torchscience.hypergeometric_2_f_1(a, b, c, z)
