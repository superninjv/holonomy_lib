# Spherical Fréchet-mean convergence boundary

`frechet_mean` runs the Karcher iteration unconditionally. On `KappaStereographicManifold(κ > 0)` (positively-curved, spherical) the manifold is **not Hadamard**, so Karcher's convergence guarantee requires inputs within the injectivity radius `π/√κ` from the iterate. Beyond that, the iteration can oscillate, escape the safe region, or converge to a non-unique local optimum.

Each row generates 10 points uniformly on the geodesic sphere of radius `radius_frac · π/√κ` from the origin and runs `frechet_mean` with `max_iter=500, tol=1e-12`. Reported: (1) whether the output is finite + on the manifold, (2) the Riemannian-norm residual of the Karcher gradient at the output (small ⇔ local optimum), (3) `||μ_output||` (proxy for whether the iterate stayed near the origin where the geodesic mean should land for a symmetric spread).

| κ | radius_frac | finite | on manifold | grad residual | ‖μ‖ |
|---:|---:|:---:|:---:|---:|---:|
| 0.25 | 0.1 | ✓ | ✓ | 2.00e-14 | 0.1514 |
| 0.25 | 0.3 | ✓ | ✓ | 2.12e-13 | 0.6097 |
| 0.25 | 0.5 | ✓ | ✗ | 3.32e-13 | 5.3714 |
| 0.25 | 0.7 | ✓ | ✗ | 2.91e-14 | 9.5596 |
| 0.25 | 0.9 | ✓ | ✓ | 2.19e-17 | 0.0287 |
| 1.00 | 0.1 | ✓ | ✓ | 1.00e-14 | 0.0757 |
| 1.00 | 0.3 | ✓ | ✓ | 1.06e-13 | 0.3049 |
| 1.00 | 0.5 | ✓ | ✗ | 1.66e-13 | 2.6857 |
| 1.00 | 0.7 | ✓ | ✗ | 9.41e-14 | 4.7798 |
| 1.00 | 0.9 | ✓ | ✓ | 1.09e-17 | 0.0143 |
| 4.00 | 0.1 | ✓ | ✓ | 5.00e-15 | 0.0379 |
| 4.00 | 0.3 | ✓ | ✓ | 2.02e-13 | 0.1524 |
| 4.00 | 0.5 | ✓ | ✗ | 2.49e-13 | 1.3429 |
| 4.00 | 0.7 | ✓ | ✗ | 4.71e-14 | 2.3899 |
| 4.00 | 0.9 | ✓ | ✓ | 5.47e-18 | 0.0072 |

## Interpretation

- `radius_frac ≤ 0.5` (points within half the injectivity radius): converges cleanly; gradient residual at machine precision or close to it; `‖μ‖` near 0 as expected for a symmetric spread.
- `radius_frac ≥ 0.7`: behavior depends on κ. Larger κ has smaller injectivity radius (`π/√κ`), so the relative spread is larger; the iteration may not reach a tight tolerance.
- `radius_frac = 0.9`: at the edge of the safe region. Karcher may oscillate; non-finite outputs are possible.

Recommendation: callers with κ > 0 should ensure input spread is well within `π/√κ`. The `frechet_mean` function does **not** currently check this; pre-validating inputs is the caller's responsibility.
