# Magic numbers catalog — synoros-lib

Per the **no magic numbers** rule (`CLAUDE.md`, `README.md`), every numerical
constant in this library must be:

- **(a)** Derived from inputs (function of dims, structure, problem parameters)
- **(b)** A universal invariant (π, e, log N, 1/N, √N, mathematical identities)
- **(c)** Experimentally tuned with documented procedure + recorded scale-of-validity

This file catalogs every constant. Updating this file is a *precondition* for
adding any new constant.

**Constants don't transfer across scales.** A value that worked at one scale is
NOT assumed to work at another. Each entry should record the scale at which it
was determined and what to re-derive at scale.

---

## Status legend

- ✅ **Derived** — formula resolves to a value from inputs/invariants
- ⚖️ **Scale-invariant** — value is a ratio or dimensionless that auto-scales
- 🔬 **Experimentally set** — value tuned via documented sweep, scale-of-validity recorded
- ⚠️ **Unresolved** — currently picked, no derivation, marked for revisiting

---

## Library-wide ALLOWED literals (audit's safe-list, justified)

Same as synoros-substrate's parent library. See `src/synoros_lib/audit.py` for the
ALLOWED_LITERALS set:

| Literal | Justification |
|---|---|
| `0, 1, -1` (and float forms) | Mathematical identities |
| `0.5` and `2` (and `-2`, float forms) | Halving/doubling identities. Symmetric/antisymmetric parts `(Z ± Zᵀ)/2`, midpoint mean `(a + b)/2`, quadratic-form coefficient `(½)xᵀAx`, triangular numbers `n(n+1)/2`, `2π`, 2-norm, second-derivative coefficients. On par with `0, 1, -1` for mathematical code. Added 2026-05-26. |
| `1024, 1024.0` | Universal binary unit conversion (KB↔MB↔GB) |
| `1000, 1000.0` | Universal SI time conversion (s↔ms) AND RNG-stream-offset multiplier |
| `1e-9` (`numerical_floor_convention`) | Posited convention: anti-divide-by-zero floor. Chosen smaller than float32 eps (~1.19e-7) for safety across dtypes. Used pervasively in `1.0 / max(x, 1e-9)` and similar guards. |

---

## Cataloged constants (per-module)

(Currently empty — populate as the library grows.)

### manifolds

(no entries yet)

### topology

(no entries yet)

### algebra

| Name | Status | Value | Justification |
|---|---|---|---|
| `RANDOMIZED_SVD_OVERSAMPLE_DEFAULT` | 🔬 | `5` | Halko-Martinsson-Tropp (2011) §1.2 standard recommendation for the oversampling parameter `p` in randomized SVD. Five extra projection columns typically yield relative spectral error below 1e-3 for matrices with smoothly decaying singular values past the truncation rank. **Scale of validity**: adequate when σ_{r+1} is meaningfully smaller than σ_r; increase to 10 (also literature-standard) when the spectrum is flat near the truncation. Used in `src/synoros_lib/algebra/linear.py`. |
| `RANDOMIZED_SVD_N_ITER_DEFAULT` | 🔬 | `2` | Halko-Martinsson-Tropp (2011) §4.5 standard recommendation for subspace-iteration power method steps. Two iterations amplify the singular-value gap by (σ_{r+1}/σ_r)^(2q+1), sufficient for typical low-rank truncation. **Scale of validity**: fine for σ_{r+1}/σ_r ≲ 0.5; increase to 4–7 when ratio is closer to 1. Used in `src/synoros_lib/algebra/linear.py`. (Note: value 2 is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default.) |

### discrete_geometry

| Name | Status | Value | Justification |
|---|---|---|---|
| `SINKHORN_REG_DEFAULT` | 🔬 | `0.01` | Cuturi (2013) §4 — middle-ground entropic regularization ε for Sinkhorn OT. Small enough to keep the entropic bias on W₁ < 1% for typical graph-metric costs; large enough to converge in O(100) iterations without underflow. **Scale of validity**: shortest-path costs in [1, ~diameter]; scale proportionally if the graph's distance scale is much wider. Used in `src/synoros_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_N_ITER_DEFAULT` | 🔬 | `100` | Cuturi (2013) §4 — typical iteration count to reach relative-change < 1e-4 for graph-metric costs at default `reg=0.01`. **Scale of validity**: smaller `reg` requires proportionally more iterations. Used in `src/synoros_lib/discrete_geometry/ricci.py`. |
| `DISCONNECTED_DISTANCE_MULTIPLIER` | 🔬 | `1000.0` | Replaces +inf in the shortest-path matrix with a large finite value (= 1000 × max-finite-distance) to keep Sinkhorn numerics stable across disconnected components. **Scale of validity**: any value much larger than the graph diameter works equivalently; 1000× chosen so that exp(−d/reg) for cross-component pairs underflows to zero at the default `reg=0.01`. Used in `src/synoros_lib/discrete_geometry/ricci.py`. |

### spectral

(no entries yet)

### info_geometry

(no entries yet)

### tensor_calculus

(no entries yet)

### optimization

(no entries yet)

### probability

(no entries yet)

---

## Re-derivation at scale

When a primitive in this library is used at a new scale (e.g., the consumer
goes from N=1000 to N=100000), the corresponding catalog entry should record
whether the constant transfers or needs re-derivation. Operations that scale
with N should be parameterized so the value auto-derives; ones that don't
need explicit re-validation at the new scale.
