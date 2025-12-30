# torchscience

PyTorch operators for mathematics, science, and engineering.

## Installation

```bash
git clone https://github.com/torchscience/torchscience.git
cd torchscience
pip install -e ".[dev]"
```

## Guarantees

torchscience is intended to be a foundational library used in downstream 
optimization and machine learning tasks. To support that use case, we aim to 
provide a small set of clear, testable guarantees about operator behavior.

### Automatic differentiation

For any operator that is differentiable on its stated domain:

- **Autograd support is provided.** The operator participates in PyTorch autograd with a defined backward.
- **Gradients are stable and validated.** Gradients are tested for correctness (e.g., via numerical checks and regression tests) and exercised in common optimization regimes to catch pathological behavior.
- **Analytical gradients are preferred.** When closed-form derivatives are available, we implement them directly rather than relying on finite-difference approximations.
- **Non-smooth points are documented.** If an operator is non-differentiable at certain points, or has domain restrictions, those are documented and the backward follows the documented convention.

torchscience is not a machine-learning package, but it is designed to be a reliable substrate for packages that are.

### Complex numbers

When an operator has a mathematically meaningful extension to the complex plane:

- **Complex implementations are provided.** The operator supports complex dtypes with behavior consistent with the documented mathematical definition.
- **Branch cuts and conventions are documented.** Any branch cuts, principal value conventions, and discontinuities are explicitly documented.
- **Complex gradients are supported where applicable.** When a complex-valued gradient is well-defined under PyTorch’s autograd conventions, we provide it and test it.

### PyTorch surfaces that may evolve

Some PyTorch APIs are still evolving upstream. torchscience supports these areas, but you should expect occasional adjustments as PyTorch changes semantics or surface area:

- Quantization
- Sparse tensors (`torch.sparse`)
- Masked tensors (`torch.masked`)

When upstream changes require updates, we will document behavior changes in release notes and, when practical, provide migration guidance.
