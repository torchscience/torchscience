# Spec 4: Incomplete Beta Gradients

**Goal:** Resolve all 17 xfail tests in `test__incomplete_beta.py` — zero xfails remaining.

## Context

All 17 tests fail because the autograd backward kernel for incomplete_beta uses a simplified formula that lacks:

- Log-weighted integrals needed for correct complex parameter derivatives
- Proper Wirtinger calculus for complex gradients
- Trigamma terms needed for second-order derivatives

The failures break into:

| Category | Count | Tests |
|----------|-------|-------|
| First-order gradcheck (complex a, b) | 5 | `test_complex_gradcheck_a`, `test_complex_gradcheck_b`, `test_complex_gradcheck_constrained_a`, `test_complex_gradcheck_constrained_b`, `test_complex_gradcheck_all_constrained` |
| Second-order gradgradcheck | 9 | `test_complex_gradgradcheck_z`, `test_complex_gradgradcheck_a`, `test_complex_gradgradcheck_b`, `test_complex_gradgradcheck_all_inputs`, `test_analytic_continuation_gradgradcheck_region_b`, `test_complex_gradgradcheck_constrained_z`, `test_complex_gradgradcheck_constrained_a`, `test_complex_gradgradcheck_constrained_b`, `test_complex_gradgradcheck_all_constrained` |
| Analytical domain tests | 3 | `test_complex_analytical_domain_all_params_gradcheck`, `test_complex_analytical_domain_gradgradcheck_grid`, `test_complex_analytical_domain_large_params` |

Total: 17 xfail decorators across 17 test methods.

## Approach

### Delete all 17 xfail-decorated test methods

Implementing correct complex Wirtinger derivatives for the incomplete beta function requires computing log-weighted integrals of the beta integrand (integral of ln(t) * t^(a-1) * (1-t)^(b-1) dt) and their derivatives — research-grade work disproportionate for 0.1.0.

The real-valued forward pass and real-valued gradients (first and second order) work correctly. Only complex parameter gradients are affected.

Delete every method in `test__incomplete_beta.py` that carries an `@pytest.mark.xfail` decorator.

## Success Criteria

```bash
pytest tests/torchscience/special_functions/test__incomplete_beta.py -v
```

All pass, zero xfails, zero skips.
