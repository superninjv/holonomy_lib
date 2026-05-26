# Magic numbers catalog — holonomy-lib

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

Same as synoros-substrate's parent library. See `src/holonomy_lib/audit.py` for the
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

| Name | Status | Value | Justification |
|---|---|---|---|
| `RETRACTION_RANDOMIZED_THRESHOLD` | 🔬 | `0.25` | `FixedRankManifold.retraction` switches from exact to randomized SVD when `r ≤ 0.25 · min(m, n)`. **Scale of validity**: derived from CPU benchmarks at 64-1024-square sizes (`notes/benchmark_baseline.md` 2026-05-27) — randomized SVD is strictly faster than full-and-truncate below this ratio, by 8× at r/min ≈ 12% and 37× at r/min ≈ 3%. At ratios above 25% the constant factors flip and exact wins back. Re-derive when GPU benchmarks land. Used in `src/holonomy_lib/manifolds/fixed_rank.py`. |

### topology

(no entries yet)

### algebra

| Name | Status | Value | Justification |
|---|---|---|---|
| `RANDOMIZED_SVD_OVERSAMPLE_DEFAULT` | 🔬 | `5` | Halko-Martinsson-Tropp (2011) §1.2 standard recommendation for the oversampling parameter `p` in randomized SVD. Five extra projection columns typically yield relative spectral error below 1e-3 for matrices with smoothly decaying singular values past the truncation rank. **Scale of validity**: adequate when σ_{r+1} is meaningfully smaller than σ_r; increase to 10 (also literature-standard) when the spectrum is flat near the truncation. Used in `src/holonomy_lib/algebra/linear.py`. |
| `RANDOMIZED_SVD_N_ITER_DEFAULT` | 🔬 | `2` | Halko-Martinsson-Tropp (2011) §4.5 standard recommendation for subspace-iteration power method steps. Two iterations amplify the singular-value gap by (σ_{r+1}/σ_r)^(2q+1), sufficient for typical low-rank truncation. **Scale of validity**: fine for σ_{r+1}/σ_r ≲ 0.5; increase to 4–7 when ratio is closer to 1. Used in `src/holonomy_lib/algebra/linear.py`. (Note: value 2 is also in `ALLOWED_LITERALS` as a doubling identity; this entry documents the *semantic* role of the default.) |

### provenance

| Name | Status | Value | Justification |
|---|---|---|---|
| `HEX_PREFIX_LEN` | 🔬 | `16` | Length (in hex chars) of the truncated sha256 prefix used as provenance IDs. 16 chars = 64 bits ≈ collision-free up to ~2³² operations by the birthday bound. **Scale of validity**: fine for interactive mech-interp sessions and most batch runs. Bump to 32 chars (128 bits) for million-op production traces. Used in `src/holonomy_lib/provenance/protocol.py`. |

### discrete_geometry

| Name | Status | Value | Justification |
|---|---|---|---|
| `SINKHORN_REG_DEFAULT` | 🔬 | `0.01` | Cuturi (2013) §4 — middle-ground entropic regularization ε for Sinkhorn OT. Small enough to keep the entropic bias on W₁ < 1% for typical graph-metric costs; large enough to converge in O(100) iterations without underflow. **Scale of validity**: shortest-path costs in [1, ~diameter]; scale proportionally if the graph's distance scale is much wider. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_N_ITER_DEFAULT` | 🔬 | `100` | Cuturi (2013) §4 — typical iteration count to reach relative-change < 1e-4 for graph-metric costs at default `reg=0.01`. **Scale of validity**: smaller `reg` requires proportionally more iterations. Note: 100 iter often lands inside the convergence-plateau region for wide-cost-range inputs (asymmetry ~1.75e-2); crank to ~2000 with `tol=1e-12` for machine-precision symmetry. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TOL_DEFAULT` | ⚖️ | `1e-9` | Log-domain Sinkhorn early-stop tolerance — iteration stops once max\|Δlog_u\| < tol. Reuses the library's `numerical_floor_convention`: well below typical use cases' noise floor, but not chasing machine-precision residuals. **Scale of validity**: dimensionless (acts on dual variables); does not need re-derivation across input sizes. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_TILE_DEFAULT` | 🔬 | `256` | Pairs processed simultaneously inside the Sinkhorn loop. Peak inner-broadcast memory = O(tile · n² · element_size); at tile=256 and float64 that caps at ~16 MB for n ≤ 256. **Scale of validity**: balances Python-loop overhead against memory pressure on the inner `(tile, n, n)` broadcast. Re-derive when running on a memory-constrained GPU or at n > ~512. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `SINKHORN_SYNC_EVERY_DEFAULT` | ⚖️ | `8` | How often to host-sync Sinkhorn's convergence check (`.item()` on the delta tensor). Per-iter syncs add tens of µs of kernel-queue wait on GPU; cadence=8 amortizes that cost by 8× while only delaying convergence detection by at most 7 extra iterations. **Scale of validity**: dimensionless cadence; does not depend on input size or dtype. Cataloged 2026-05-27. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `DISCONNECTED_DISTANCE_MULTIPLIER` | 🔬 | `1000.0` | Replaces +inf in the shortest-path matrix with a large finite value (= 1000 × max-finite-distance) to keep Sinkhorn numerics stable across disconnected components. **Scale of validity**: any value much larger than the graph diameter works equivalently; 1000× chosen so that exp(−d/reg) for cross-component pairs underflows to zero at the default `reg=0.01`. Used in `src/holonomy_lib/discrete_geometry/ricci.py`. |
| `RICCI_FLOW_SURGERY_PERIOD_DEFAULT` | 🔬 | `10` | Number of Ricci-flow steps between surgery passes. Ni-Lin-Luo-Gao (2019) §3.2 use values in [10, 15] for community detection; 10 is the more aggressive choice. **Scale of validity**: smaller graphs / faster surgery → smaller period (down to 5); huge graphs may want 20+. Used in `ricci_flow_with_surgery`. |
| `RICCI_FLOW_SURGERY_THRESHOLD_DEFAULT` | 🔬 | `3.0` | Surgery cutoff multiplier — edges with weight ≥ 3.0 × initial mean weight are removed as "necks". Sia (2019) and Ni (2019) use thresholds in [2, 3]; 3.0 is the conservative end (less aggressive removal). **Scale of validity**: graphs with wider initial-weight distributions may need a larger threshold to avoid removing legitimate heavy edges. Used in `ricci_flow_with_surgery`. |

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
