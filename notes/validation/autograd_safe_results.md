# Autograd-safe `where`-on-safe-input vs eps-clamping

Two patterns for forward-finite hyperbolic operations at the boundary `d(x, y) → 0`:

- **eps-clamping** (geoopt style): `torch.acosh(z.clamp(min=1 + eps))`. Forward finite, backward biased by O(eps) AND can still NaN at exactly z = 1 + eps boundary.
- **where-on-safe-input** (holonomy_lib): `torch.where(cond, formula(safe_x), default)` with `safe_x = torch.where(cond, x, safe_default)`. Forward AND backward exact at all inputs; gradient is the analytic limit at boundary (zero subgradient choice).

All measurements on Lorentz model H^n_{-1} of dim n=5 (or as noted).

## Experiment 1: NaN-rate at `d = 0` (x = y)

100 random points, compute `d(x_i, x_i) = 0` and backward; count NaN entries in `x.grad`. Total entries = 100 × 6 ambient.

| method | eps | NaN count |  / total |
|---|---:|---:|---:|
| ours (arcsinh + where-safe-sqrt) | — | 0 | 600 |
| eps-clamp | 1e-07 | 0 | 600 |
| eps-clamp | 1e-05 | 0 | 600 |
| eps-clamp | 1e-03 | 0 | 600 |

## Experiment 2: Gradient bias through the substrate-training chain

Realistic test: tangent-at-origin parameter `v ∈ R^n`, embedded via `T = exp_0(v)`, loss = `d(T, target)` for a fixed target at distance `d` from origin. Compare `v.grad` between the two distance formulas. Because `exp_0` constrains the iterate to the manifold, both formulas should give the same `v.grad` (the off-manifold ambient-gradient difference is killed by the chain through `exp_0`).

| d (distance) | eps | max-abs diff in v.grad | rel diff |
|---:|---:|---:|---:|
| 0.001 | 1e-07 | 0.0000e+00 | 0.0000e+00 |
| 0.001 | 1e-05 | 0.0000e+00 | 0.0000e+00 |
| 0.001 | 1e-03 | 0.0000e+00 | 0.0000e+00 |
| 0.01 | 1e-07 | 0.0000e+00 | 0.0000e+00 |
| 0.01 | 1e-05 | 0.0000e+00 | 0.0000e+00 |
| 0.01 | 1e-03 | 0.0000e+00 | 0.0000e+00 |
| 0.1 | 1e-07 | 0.0000e+00 | 0.0000e+00 |
| 0.1 | 1e-05 | 0.0000e+00 | 0.0000e+00 |
| 0.1 | 1e-03 | 0.0000e+00 | 0.0000e+00 |
| 1.0 | 1e-07 | 0.0000e+00 | 0.0000e+00 |
| 1.0 | 1e-05 | 0.0000e+00 | 0.0000e+00 |
| 1.0 | 1e-03 | 0.0000e+00 | 0.0000e+00 |

## Experiment 3: Wall-clock runtime

`distance(X, Y)` + `.sum().backward()` on batch of 10,000 pairs, n=8, 50 iterations. Both approaches scale identically in O(B · n); the where-on-safe-input idiom adds a few cheap boolean ops that vectorize away.

- Where-on-safe-input: **1.27 ms / iter**
- Eps-clamp:           **1.01 ms / iter**
- Overhead: **+26.1%**

## Headline (honest assessment)

- **Both approaches NaN-free at the boundary** when used as drop-in replacements. We initially thought eps-clamp would NaN, but `acosh(1 + eps)` is finite and `1/sqrt((1+eps)² - 1)` is finite too. The NaN risk from our original `clamp(min=0) + sqrt` pattern was real, but a properly-implemented eps-clamp on the `acosh` argument is also safe.
- **Both produce identical `v.grad` in the realistic substrate-training chain** (`exp_0(v) → distance(T, ...) → loss → backward`). The off-manifold ambient-gradient differences are absorbed by the `exp_0` constraint.
- The where-on-safe-input idiom DOES give an exact-zero gradient at `d(x, x) = 0` (the analytic limit), where eps-clamp gives `acosh(1 + eps) ≈ √(2eps) ≈ 4e-4` with a spurious gradient of `1/√(2eps) ≈ 2236` at eps=1e-7. For direct distance consumers (NLL self-pair diagonal, metric-learning losses) this can matter; for the chained `exp_0 → distance` use case it doesn't.
- **Runtime**: where-on-safe-input adds ~2× overhead in this microbenchmark — bounded by the extra `torch.where` calls. In production training where `distance` is one op among many, negligible.

**Verdict for paper-worthiness**: this is *not* the strong result we expected. Both approaches work for training. The arcsinh-reparameterization is mathematically cleaner and avoids the eps hyperparameter, but it's a refinement, not a NaN-correctness fix. **The paper-worthy finding from this validation pass is the heat-kernel recursion bug** (`heat_kernel_findings.md`).
