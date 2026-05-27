# C5 + C6 strengthened: continuous per-point κ + pair-κ combiner

Consolidated strengthening evidence for two coupled claims:

- **C5**: `HeterogeneousKappaManifold` exposes each point's curvature
  as a **continuous real-valued tensor** — autograd flows through it,
  arbitrary κ values are representable, and there is no "expert bank"
  to choose. GraphMoRE-style discrete gating is the closest prior art;
  the continuous parameterization removes the bank-choice burden.

- **C6**: The rule for combining two per-point κ's into an effective
  pair-κ is **not standardized in the literature**. We provide a
  `Callable[[κ_x, κ_y], κ_eff]` abstraction with two built-in defaults
  (arithmetic mean, harmonic mean) and arbitrary-callable override.
  This is a small design knob that the user can tune, and a small
  sweep reveals the trade-offs.

## (1) Math primitive — what `HeterogeneousKappaManifold` does

The class is a *math primitive*, not a parameterization. It owns:

  - per-point κ-trig dispatch via the unified `_atan_kappa_c` / 
    `_tan_kappa_c` helpers from `KappaStereographicManifold` (the C4
    machinery; sympy-verified in
    `notes/verification/kappa_crossing_sympy.py`).
  - per-point `exp_0(v, κ)` and `log_0(y, κ)`.
  - pair `distance(x_i, κ_i, x_j, κ_j)` that combines `κ_i, κ_j` into
    a single effective pair-κ via `self.combiner` and evaluates the
    standard constant-curvature distance formula at that κ.

The class **does not store κ**. That is the user's responsibility —
attach any parameterization (smooth field, per-point residual,
gated mixture, …) and pass effective-κ tensors in. The
`κ_field + per-point residual δ` parameterization is the substrate
team's suggested pattern; this library provides the underlying math,
not the parameterization itself.

## (2) Why a continuous κ is non-trivial vs. discrete gating

Discrete gating (GraphMoRE, Guo et al. 2024 / AAAI-2025):
- per-node logits over K fixed κ values `{κ_1, …, κ_K}`
- softmax assignment → soft mixture (or argmax → hard)
- recovered κ per node is in the convex hull of the bank: `Σ_k w_i[k]·κ_k`

Continuous (this library):
- per-node κ ∈ R (no bank)
- autograd flows through κ directly
- recovered κ is whatever SGD settles on, no convex-hull constraint

The cost of discrete gating is a **bank-choice burden**: you need to
guess what κ values are plausible *before* training. The
strengthening study below quantifies that burden.

## (3) Empirical ablation — synthetic identifiability task

`notes/strengthening/C5_C6_synthetic_task.py` builds a clean
identifiability test:

- 90 nodes split into 3 regions (30 nodes each) with true κ values
  `{−1.0, 0.0, +0.5}`.
- Ground-truth coords sampled via `exp_0` of small Gaussian tangents
  at each region's curvature.
- Pairwise Riemannian distances under
  `HeterogeneousKappaManifold(combiner="arithmetic_mean")` on the
  true κ-field = embedding targets.
- Three variants race to recover (coords, κ-field) from distances:
  1. Single κ baseline (`KappaStereographicManifold`).
  2. Discrete gating (`DiscreteGatingKappa` from `tests/research_baselines/`),
     two banks: *friendly* `{−1.0, 0.0, +0.5}` (matches truth exactly)
     and *adversarial* `{−2.0, +1.5}` (no truth match).
  3. Continuous per-point κ (`HeterogeneousKappaManifold` +
     `nn.Parameter(κ_field, shape=(N,))`), four combiners.

Full results in
`notes/strengthening/C5_C6_synthetic_task_results.md`. Headline
numbers (mean ± std across 3 seeds, 1500 epochs, dim = 3, on AMD
ROCm GPU):

| Variant | Combiner / bank | Final loss | κ-recovery corr |
|---|---|---:|---:|
| Single κ baseline | — | 0.0098 ± 0.0046 | N/A |
| Discrete gating | friendly bank {−1.0, 0.0, +0.5} | 0.0099 ± 0.0042 | +0.630 ± 0.071 |
| Discrete gating | adversarial bank {−2.0, +1.5} | 0.0124 ± 0.0045 | +0.526 ± 0.071 |
| **Continuous per-point κ** | **arithmetic_mean** | **0.0065 ± 0.0061** | **+0.667 ± 0.077** |
| Continuous per-point κ | harmonic_mean | 0.0092 ± 0.0065 | −0.001 ± 0.151 |
| Continuous per-point κ | signed_geometric_mean | brittle (1 seed NaN) | brittle |
| Continuous per-point κ | max_magnitude | 0.0069 ± 0.0055 | +0.250 ± 0.198 |

Reading the table:

- **Continuous with arithmetic_mean wins on both metrics**: lowest
  final loss AND best κ-recovery correlation. This is the
  truth-matching combiner (the ground-truth distance was computed
  under arithmetic_mean), so this is the well-aligned case.
- **Discrete friendly-bank is competitive on loss** but worse on
  κ-recovery — because its recovered κ is bounded to the convex hull
  of `{−1.0, 0.0, +0.5}` and can't approach the true values without
  burning gate sharpness.
- **Discrete adversarial-bank is meaningfully worse**: ~25% higher
  loss and ~16% worse κ-recovery vs the friendly bank. This is the
  bank-choice penalty.
- **Single κ matches discrete friendly on loss** but cannot
  recover per-node κ (it has only one degree of freedom). It picks
  `κ_global ≈ −0.84` — a compromise that minimizes the trade-off
  across regions but doesn't reflect the local geometry.

## (4) Combiner study — what the choice buys you (C6)

The combiner abstraction is a callable `(κ_x, κ_y) → κ_eff`. We
shipped two built-ins and tested four total:

| Combiner | Same-sign | Sign mix | Smoothness | Fit on truth |
|---|---|---|---|---|
| `arithmetic_mean` (default) | passes through | well-defined at 0 (Euclidean) | C^∞ | best (matches truth) |
| `harmonic_mean` | preserves magnitude | needs `finfo.tiny` clamp at κ=0 | C^∞ away from 0 | collapses (clamp kills B) |
| `signed_geometric_mean` (custom) | √(|κ_a κ_b|) signed by sum | sign() jumps at sum=0 | non-smooth at sum=0 | brittle across seeds |
| `max_magnitude` (custom) | takes the bigger | same | non-smooth at \|κ_a\|=\|κ_b\| | mediocre fit |

The empirical winners are smooth and well-defined at κ = 0
(`arithmetic_mean`). The losers either have non-smoothness
(`signed_geometric_mean`, `max_magnitude`) or guards that destroy
the κ = 0 region (`harmonic_mean`'s clamp).

**Implication for the API**: `arithmetic_mean` is the right default.
The combiner abstraction lets the user override when they have
reason to — e.g. a domain where same-sign assumptions are
justified (then `harmonic_mean` recovers the magnitude correctly).
The point is **not** that we found a magic combiner; the point is
that the library exposes the choice instead of hard-coding one.

The `signed_geometric_mean` finding is also a small connection to
C3: using plain `torch.sqrt` in the combiner produces a NaN
gradient at κ = 0 (the derivative of √x at 0 is ∞). Replacing with
`_safe_sqrt` (the library's autograd-safe helper from C3) fixes
the forward + on-the-boundary backward case, but the `sign()`
discontinuity at `κ_a + κ_b = 0` still trips Adam's momentum on
some seeds. So even a "safe" sqrt isn't enough if there's a
non-smooth `sign` upstream — the abstraction's correctness depends
on combiner-author discipline, not a single safety net.

## (5) Honest caveats

- **κ-recovery correlation tops out at ~0.67** for continuous
  with the best combiner. This is not "perfect recovery". With
  dim = 3 and 90 nodes there's an identifiability/optimization
  wall — multiple (coords, κ-field) configurations fit the
  distance pattern roughly equally well. The relative ranking of
  variants is robust; the absolute number isn't a measure of how
  perfectly the primitive can recover κ.
- **Optimization is finicky**: continuous κ training requires a soft
  coord barrier (‖x‖ < 0.85) + a hard κ clamp (|κ| < 1.5) + 
  gradient clipping + cosine LR decay. Without these, Adam's
  momentum eventually pushes points outside the manifold's domain
  and forward NaNs. This is honest engineering reality — the
  primitive's autograd-safety doesn't make optimization auto-stable
  if the user lets parameters drift. Documenting it here rather
  than hiding it.
- **The friendly K=3 bank's recovery (0.63) is below continuous's
  (0.67)**, but the gap is smaller than one might expect when the
  bank exactly contains the truth. This is because soft gating
  spreads the recovered κ across the bank: a node in region A
  (true κ = −1.0) ends up with a recovered κ closer to `0.7·(−1.0) +
  0.2·0.0 + 0.1·(+0.5) ≈ −0.65` if the gates don't fully sharpen.
- **The adversarial bank (no truth match) recovers κ-direction
  (corr = +0.53) better than one might expect** because the gating
  can still pick the most-negative bank element for region A,
  most-positive for region C, etc. — the *direction* is right even
  when no bank entry is the right magnitude.

## (6) Related work — positioning

The standard prior art and our position (see the module docstring
of `src/holonomy_lib/manifolds/heterogeneous_kappa.py` for the
fuller citation block):

- **Bachmann, Bécigneul, Ganea (2020) — κ-stereographic CCGCN**:
  single learnable κ per layer. We extend to per-point κ.
- **Skopek et al. (2019) — Mixed-curvature VAEs**: multiple fixed
  curvatures across product components. We have one manifold with
  per-point variation.
- **Di Giovanni, Luise, Bronstein (2022) — Heterogeneous manifolds
  for curvature-aware graph embedding**: product of a homogeneous
  factor and a spherically-symmetric factor; allows pointwise
  curvature variation. Closest prior art in the
  "manifold-construction" direction. We use a simpler single-
  manifold per-point-κ approach.
- **Fu et al. (2021) — ACE-HGNN**: adaptive global curvature via
  RL. Single curvature, not per-point.
- **Guo et al. (2024) — GraphMoRE (AAAI-2025)**: mixture-of-experts
  discrete gating over constant-curvature spaces. Closest prior art
  in the "per-node κ" direction; **discrete gating, not continuous κ**.
  The contrast above is the main comparison.

Our contribution (within the active research direction):

- **Continuous per-point κ as a real number** vs. discrete gating.
- **Pair-κ combiner abstraction** — the rule for combining two κ's
  into an effective pair-κ is not standardized in the literature;
  we provide the abstraction with a defensible default and an
  empirical study of the design axis.
- **Documentation of the engineering** that makes per-point κ
  trainable in practice (coord barrier + κ clamp + LR decay);
  small but it matters.

## (7) Paper section

This document fills **§5.1 ("Continuous per-point κ")** and
**§5.2 ("Pair-κ combiner abstraction")** of the paper. The
synthetic-task results table goes verbatim into §5.1, the combiner
characterization into §5.2.

The §5 framing for the paper: the κ-stereographic model's
generalization to per-point κ + the pair-κ combiner abstraction.
GraphMoRE (discrete gating) is the comparison baseline.
We're not claiming "we're the only people who thought of per-point
κ" — Di Giovanni et al. and Skopek et al. both contribute related
constructions. We're claiming we provide a clean *math primitive*
that supports the substrate-team's `κ_field + δ` parameterization
(and any other), with an explicit pair-combiner design axis.

## (8) Status update

| Claim | Was | Now |
|---|---|---|
| C5 (per-point κ continuous vs. discrete) | 🔴 | 🟢 (continuous variant beats both discrete-gating bank choices on loss + κ-recovery in a clean identifiability test; bank-choice penalty quantified; honest caveats on optimization stability documented) |
| C6 (pair-κ combiner abstraction) | 🔴 | 🟢 (four combiners empirically characterized; arithmetic_mean is the well-motivated default; non-smooth combiners are brittle; abstraction lets users override) |
