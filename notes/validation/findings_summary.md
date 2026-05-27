# Validation pass — summary of findings

Three validation experiments on the hyperbolic extension:

| # | Subject | Finding | Severity |
|---|---|---|---|
| 1 | Heat-kernel recursion for n ≥ 5 | **Bug found and fixed** | 🔴 critical |
| 2 | Spherical Frechet mean boundary | Known limitation characterized; fail-loud check added | 🟡 medium |
| 3 | Autograd-safe idiom vs eps-clamping | Refinement, not a correctness fix | 🟢 low |

Detailed reports in `notes/validation/*.md`.

---

## Paper-worthy items

### Strong: heat-kernel recursion bug + general-n differentiable implementation

**The novel contribution.** Across the PyTorch hyperbolic-ML
ecosystem (geoopt, geomstats, pymanopt, hyperbolic_clib, ...), no
library implements the hyperbolic heat kernel at all, much less in
a fully-differentiable general-`n` form. Our implementation closes
that gap.

The bug we found is also pedagogically valuable: the "simple"
Grigor'yan-Noguchi recursion `k^{n+2} = -(2π sinh r)^{-1} · ∂_r k^n`
that appears in some references is **wrong** — it's missing a
spectral-shift factor `exp(-n·t)`. The correct form is

  k^{n+2}(t, r) = -exp(-n·t) / (2π·sinh r) · ∂_r k^n(t, r).

We caught this via a simple validation strategy (heat-equation
PDE residual + probability-mass integration) that any future
manifold-PDE implementation could reuse.

**Paper potential**: workshop paper / technical report. ~6-8 pages.
Contribution: (a) general-n differentiable hyperbolic heat kernel
in PyTorch, (b) bug-correction + the validation methodology that
caught it, (c) end-to-end autograd demonstration on a downstream
ML task (we'd need to design one).

**Open questions** before submission:
- Even n ≥ 4 implementation (currently raises `NotImplementedError`).
- Characterize the safe `(n, t, r)` envelope where the
  autograd-chained recursion stays accurate. We see degradation at
  large n × large t (n=9, t=5.0 gives NaN).
- Demonstrate downstream utility — e.g. hyperbolic graph kernel
  for node classification on a hierarchical dataset, comparing to
  Euclidean baseline.

### Medium: arcsinh reparameterization for autograd-safe hyperbolic geometry

Our `_safe_sqrt` / `_safe_sinhc` / `_safe_arcsinhc` helpers are
clean implementations of an established idiom (the
`where(cond, formula(safe_x), default)` pattern). They produce
exact analytic-limit gradients at boundaries; eps-clamping (the
geoopt-style approach) produces finite-but-biased gradients at
boundaries.

**Critical caveat**: in the realistic training chain
`v → exp_0(v) → distance(T, target) → loss`, the constraint
through `exp_0` makes both approaches produce identical `v.grad`.
The where-on-safe idiom is only meaningfully better when distance
is consumed *directly* (without the `exp_0` constraint chain).

**Paper potential**: best-practices note or section in a larger
paper. Not a standalone contribution.

### Low: validation strategy itself

The PDE-residual + mass-normalization validation pattern (which
caught the heat-kernel bug) is broadly applicable to any
manifold-PDE primitive. The technique isn't novel — it's a
standard numerical-analysis sanity check — but applying it to
ML-library primitives where the "is this right?" question is
otherwise hand-waved is worth a short pedagogical writeup.

---

## What we fixed in this pass

- ✅ Heat-kernel recursion bug (n ≥ 5): added `exp(-n·t)` factor
- ✅ Even-n heat kernel: explicit `NotImplementedError` (was returning incorrect values)
- ✅ Spherical Frechet mean: fail-loud `RuntimeError` when the iteration escapes the domain
- ✅ Frechet convergence-norm: at OLD μ, not NEW (caught in earlier reviewer pass)

## What remains open

- Even n ≥ 4 heat kernel implementation (math derivation needed)
- (n=9, t=5.0) numerical-blowup regime in the recursion
- Demonstrating downstream ML utility for the heat kernel (paper experiments)
- Spherical Frechet mean — currently fails-loud on domain escape; an alternative
  approach would be to *project* the iterate back onto the manifold at each step
  (would converge for any input, but to a possibly-non-unique minimum)

## What this means for downstream consumers

- **Use the heat kernel at n ∈ {1, 2, 3, 5, 7, 9}** for now; even n ≥ 4 raises.
- **Use Lorentz / KappaStereographic (κ < 0) freely** — these are well-validated.
- **For KappaStereographic with κ > 0** (spherical), keep input spread well within
  the injectivity radius `π/√κ`; `frechet_mean` will raise if you don't.
- **Tangent-at-origin training pattern** (`T = exp_0(v); loss = d(T, target)`) is
  the recommended approach — autograd-safe across all manifolds.
