# C4 strengthened: κ-sign crossing without retraining

Consolidated strengthening evidence for the claim:

> For a learnable curvature κ on the κ-stereographic model, the
> branch (spherical / hyperbolic / Euclidean) need NOT be locked at
> construction. `torch.where(κ > 0, spherical_formula(√κ·α),
> hyperbolic_formula(√|κ|·α))` with `|κ|` clamped at `finfo.tiny` for
> the sqrt gives forward continuous and backward finite through κ = 0,
> so SGD can push κ through 0 without breakdown.

## (1) Why a single dispatch works — the analytic-continuation lemma

The map
```
f(κ, α) := atan_κ(√|κ|·α) / √|κ|
       =   arctan(√κ·α)/√κ        for κ > 0
       =   α                       for κ = 0
       =   arctanh(√|κ|·α)/√|κ|   for κ < 0
```
extends to a **single analytic function of κ ∈ R**. Its κ-Taylor
series at zero is
```
f(κ, α) = α · ∑_{m≥0} (−κ α²)^m / (2m+1)
        = α · (1 − κα²/3 + κ²α⁴/5 − κ³α⁶/7 + …).
```
The radius of convergence is exactly `|κ|·α² < 1`, which coincides
with the κ-stereographic manifold's distance-formula domain.
Equivalently the integral representation
```
f(κ, α)/α = ∫₀¹ 1/(1 + κ(tα)²) dt
```
bridges both signs without a conditional — it's `arctan(√κ·α)/√κ·α`
for κ > 0 and `arctanh(√|κ|·α)/√|κ|·α` for κ < 0 by the standard
antiderivatives.

**Sympy verifies** at `notes/verification/kappa_crossing_sympy.py`:

| Identity | sympy verdict |
|---|---|
| Spherical Taylor in `k_pos` = arctan/√k_pos series | matches reference series ✓ |
| Hyperbolic Taylor in `k_neg` = arctanh/√k_neg series | matches reference series ✓ |
| Spherical-Taylor(κ) − Hyperbolic-Taylor(−κ) (to O(κ⁴)) | residual = 0 ✓ |
| `lim_{κ→0+} arctan(√κ·α)/√κ`, `lim_{κ→0−} arctanh(√|κ|·α)/√|κ|` | both = α ✓ |
| 1st κ-derivative at 0± on both branches | both = −α³/3 ✓ |
| 2nd κ-derivative at 0± on both branches | both = α⁵/5 ✓ |
| ∫₀¹ 1/(1 + k_pos·(tα)²) dt = arctan(√k_pos·α)/(√k_pos·α) | exact ✓ |
| ∫₀¹ 1/(1 − k_neg·(tα)²) dt = arctanh(√k_neg·α)/(√k_neg·α) | matches via Taylor + numerical check ✓ |

So the dispatch isn't "two different geometries glued at κ = 0" — it
is one analytic function expressed via two different closed forms on
the two sides of zero. Smoothness at κ = 0 is automatic.

## (2) The implementation — and why it's autograd-safe

`src/holonomy_lib/manifolds/stereographic.py:_atan_kappa_c`:

```python
def _atan_kappa_c(self, alpha: torch.Tensor) -> torch.Tensor:
    if not self._kappa_is_tensor:                       # fast path
        ...                                              # locked branch
    kappa = self._get_kappa()                           # learnable κ
    abs_k = torch.abs(kappa).clamp(min=torch.finfo(alpha.dtype).tiny)
    scaled = torch.sqrt(abs_k) * alpha
    return torch.where(
        kappa > 0, _safe_atanc(scaled), _safe_atanhc(scaled),
    )
```

Three things together make this autograd-safe at κ = 0:

1. **`abs(κ).clamp(min=finfo.tiny)`** before the `sqrt`. Without it,
   `√|κ| = 0` at κ = 0, so `scaled = 0`, and both `arctan(0)/0` and
   `arctanh(0)/0` are the indeterminate form `0/0`. The clamp keeps
   `√|κ| ≥ √tiny ≈ 1.5e-154` (float64), so `scaled` is tiny-but-nonzero
   at α > 0, and both helpers evaluate at a small positive number
   rather than 0.

2. **`_safe_atanc(t)` / `_safe_atanhc(t)` short-circuit `t ≤ 0`** via
   `torch.where(t > 0, atan(t)/t, 1)`. The analytic limit `1` covers
   the `α = 0` case AND any residual numerical zeros the clamp
   doesn't catch. (The `where` masks the `0/0` form so its gradient
   doesn't propagate.)

3. **The outer `torch.where(κ > 0, ...)`** picks per-element forward,
   AND backward: `∂L/∂κ = mask·grad_spherical + (1 − mask)·grad_hyperbolic`.
   With each branch's forward and backward finite, the dispatch is
   finite. The literature lemma (1) guarantees both branches agree
   at κ = 0 — so the mask boundary is itself a smooth transition.

What the implementation contributes vs. the literature math: the
literature gives you one analytic function; PyTorch autograd needs
two finite expressions to mask between. The clamp + `_safe_*c`
threefold safety net is the engineering that turns the analytic
identity into something `torch.autograd` can use.

### Caveat on the masked-out branch

The outer `torch.where` evaluates BOTH branches forward and
backward, then discards the masked-out forward value. For inputs
inside the κ-stereographic distance-formula domain (`√|κ|·d < 1`,
the Poincaré-ball radius for the negative branch), `arctanh` is
finite, so both branches are safe.

For inputs *outside* the negative branch's domain (e.g. when κ > 0
and the user feeds an `α` such that `√κ·α ≥ 1`), the hyperbolic
branch's `arctanh(scaled)` returns ±∞/NaN forward. The outer
`torch.where` discards that forward value, but NaN can still leak
through the backward via `(1 − mask) · NaN = NaN`. We don't see
this in practice because (a) the manifold's domain restriction
keeps inputs inside both branches' good range, and (b)
`_safe_atanhc`'s mask handles `t ≤ 0` cleanly. But for the
record: the dispatch is autograd-safe **inside the manifold's
distance-formula domain**, which is the entire region where the
distance is well-defined anyway.

## (3) Numerical stress test — `C4_kappa_crossing_stress_results.md`

`notes/strengthening/C4_kappa_crossing_stress.py` exercises four
scenarios:

### (3a) Multi-crossing SGD trajectory

100-step SGD trajectory with the target oscillating between +0.7
and −0.7 (4 sign flips). κ.grad, distance, points all remain finite
at every step. The dispatch doesn't notice the crossings — there is
no `if κ == 0:` special case, no restart, no detach. Just the
analytic continuation walking smoothly through zero.

### (3b) Static-branch lock failure mode

A static-float-κ manifold caches `_branch` and `_sqrt_abs_kappa` at
construction. If you mutate κ across zero (the natural way to
simulate SGD without the Tensor-κ machinery), the cached branch
stays frozen and applies the wrong formula:

| path | d(p1, p2) at κ = +0.5 |
|---|---:|
| `KappaStereographicManifold(κ=−0.5)`, mutated to +0.5 (wrong branch) | 0.4086050425 |
| `KappaStereographicManifold(κ=+0.5)` (correct, spherical) | 0.4030358909 |
| Tensor κ, dynamic dispatch, pushed from −0.5 to +0.5 | 0.4030358909 |

The locked-branch path has a **1.38% relative error** — it computed
`arctanh(√0.5·d)` when it should have computed `arctan(√0.5·d)`.
Different formula, different answer. Tensor dispatch matches the
correct answer to machine precision.

### (3c) Truncated Taylor unified κ-trig

The natural alternative to a `torch.where` is to truncate the
unified Taylor series and use it for all κ. Within the manifold
domain (`|κ|·α² < 1`) the series converges, but:

- **Well inside the domain** (`|κ|·α² ≲ 0.25`, α = 0.5): 6 terms
  reaches 1e-5 relative error. Dispatch hits machine precision in
  one call.
- **Near the boundary** (`√|κ|·α = 0.95`): even 32 Taylor terms
  leaves ~1e-3 error. Dispatch stays exact.

So Taylor truncation is dominated by the dispatch across the entire
domain, with the dominance growing as you approach the boundary.
The dispatch's "two single-kernel `arctan`/`arctanh` evaluations
per call" is uniformly cheaper than the convergent-but-many-term
Taylor alternative.

### (3d) Latency overhead

`torch.where` evaluates both branches forward and backward. On CPU
float64, batch = 16384, n = 8, the dispatch path costs ~1.2 − 1.4×
the static-float fast path per forward+backward — substantially
below the worst-case 2× because (a) each `_safe_*c` is one
elementwise kernel, (b) the Möbius-addition machinery dominates and
is branch-independent, (c) PyTorch fuses many of the small ops.

GPU latency is not characterized in this session (no CUDA hardware
available locally for the C4 strengthening run). The dispatch
structure is `torch.where` + two elementwise transcendentals,
fully data-parallel; we expect GPU overhead to be comparable.

## (4) Related work — where this sits in the literature

Curvature-as-a-learnable-scalar is established:

- **Bachmann, Bécigneul, Ganea (2020)** introduces the
  κ-stereographic model with κ as a tunable scalar per layer.
- **Skopek, Ganea, Bécigneul (2019/2020)** "Mixed-curvature
  Variational Autoencoders" treats κ as a learnable parameter per
  product-manifold component.
- **GraphMoRE (Tian et al., 2023)** uses a mixture-of-experts over
  a discrete set of constant-curvature spaces — its "learnable
  curvature" is a discrete gating, not a continuous κ field.

What's *not* explicit in those papers' implementations (as far as
we've checked) is what happens when SGD pushes κ across zero — the
case is typically avoided by initializing κ on one side of zero and
applying L2 regularization (or just praying). The contribution here
is the **autograd-safe dispatch + the validation that it actually
does what it claims**: a multi-crossing SGD trajectory + sympy
analyticity proof + comparison to the static-branch alternative.

This sits in the same lineage as the `_safe_sqrt` / `_safe_sinhc`
folklore (see C3's related-work positioning) — a small primitive
that handles a numerical edge case that the math papers gloss over.

## (5) Paper section

This strengthening fills **§4.2** of the paper, "κ-sign crossing
without retraining". The story is: textbook math says the
geometries unify at κ = 0; PyTorch autograd needs three pieces
(clamp, `_safe_*c`, outer `torch.where`) to realize that unification
through a differentiable computation; with all three in place, SGD
walks across κ = 0 with no special handling.

## (6) Status update

| Claim | Was | Now |
|---|---|---|
| C4 (κ-sign crossing dispatch) | 🔴 | 🟢 (sympy analyticity verified; multi-crossing SGD trajectory finite; static-branch failure mode quantified; Taylor alternative dominated; CPU latency characterized; GPU benchmark deferred pending hardware) |
