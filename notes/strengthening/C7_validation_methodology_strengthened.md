# C7 strengthened: reference-free dual-check validation methodology

Consolidated strengthening evidence for the claim:

> For numerical implementations of operator-semigroup primitives
> (heat kernels, diffusion operators, geodesic flows), two
> *complementary, reference-free* consistency checks — an
> operator-equation residual and a conserved-quantity check —
> form a correctness gate that catches errors the standard
> "compare to a closed form at a few points" approach misses.

The claim was previously demonstrated *once* (catching the C1
hyperbolic-heat-kernel spectral-shift bug). A methodology resting on
a single instance is an anecdote. This document establishes it as a
method by (a) abstracting the two checks, (b) demonstrating they
generalize to a structurally different primitive, and (c) showing
the two checks are genuinely complementary — neither suffices alone.

## (1) The methodology, abstractly

Let `S_t` be an operator semigroup (the solution operator of an
evolution equation). Two checks:

**Check 1 — operator-equation residual.** `S_t` satisfies a defining
evolution equation `∂_t S = 𝒢 S` for a generator `𝒢`. Evaluate the
residual `∂_t S − 𝒢 S` by finite-difference in `t` and report it
relative to `‖𝒢 S‖`. A correct implementation sits at the
finite-difference truncation floor.

**Check 2 — conserved-quantity check.** The semigroup conserves a
known quantity `Q` (`Q(S_t x) = Q(x)` for all `t`). Evaluate the
deviation `Q(S_t x) − Q(x)`. A correct implementation sits at machine
precision.

Both checks are **reference-free**: they require no known-correct
`S_t` to compare against — only the object's own defining equation
and conservation law. This is what makes them useful for a *new*
primitive, where no reference exists.

## (2) Two instances

| | Hyperbolic heat kernel (C1) | Graph heat kernel (new) |
|---|---|---|
| Object | `k_t(r)` on H^n (continuous) | `K_t = exp(−t·L)` on a graph (discrete) |
| Evolution eq. | radial heat PDE `∂_t k = Δ_radial k` | linear ODE `dK/dt = −L·K` |
| Generator 𝒢 | `Δ_radial = ∂_r² + (n−1)coth(r)∂_r` | `−L` |
| Conserved Q | probability mass `∫_{H^n} k_t dV = 1` | row mass `K_t·1 = 1` (since `L·1 = 0`) |
| Residual tool | central FD in t and r | central FD in t |
| Mass tool | Gauss–Legendre quadrature | exact row sum |

- **C1**: `notes/validation/heat_kernel_validation.py` +
  `heat_kernel_results.md`. The PDE residual flagged the naive
  recursion (spectral-shift factor dropped); the residual was off by
  exactly the missing `exp(n·t)`.
- **Graph**: `notes/validation/graph_heat_kernel_validation.py` +
  `graph_heat_kernel_results.md`. On a 24-node random graph
  (combinatorial Laplacian), the correct kernel passes both checks
  (ODE residual ~1e-6 at the FD floor; mass deviation ~1e-16). Three
  negative controls below confirm the checks fire on corruption.

The methodology transfers across the continuous→discrete divide
without modification — only the concrete tools (quadrature vs. row
sum) change.

## (3) The key result — the two checks are complementary

A single check has blind spots. The graph-kernel experiment makes
this precise with three corruptions:

| Corruption | Operator residual | Mass conservation |
|---|---|---|
| (B) spectral-shift `K → exp(α·t)·K` | fires (grows with t) | fires — deviation = `exp(α·t)−1` exactly |
| (C) constant rescale `K → c·K` | **blind** (FD floor for all c) | fires — deviation = `\|c−1\|` |
| (D) mass-preserving `K → K + ε·L'` | fires (∝ ε) | **blind** (machine precision for all ε) |

The two blind spots are structural, not incidental:

- **(C) constant rescale.** A scalar multiple of a solution to the
  *homogeneous* equation `dK/dt = −L·K` is still a solution:
  `d(c·K)/dt + L·(c·K) = c·(dK/dt + L·K) = 0`. The residual *cannot*
  detect a constant rescale — it's in the null space of the check.
  Mass conservation catches it immediately (`c·K·1 = c·1`).

- **(D) mass-preserving perturbation.** Adding `ε·L'` (another
  Laplacian, so `L'·1 = 0`) leaves every row sum unchanged: mass
  conservation is blind. But `K + ε·L'` no longer solves the heat ODE
  for `L`, so the residual fires linearly in `ε`.

The C1 spectral-shift bug (B) happens to be caught by *both* — which
is why a single check was enough to *find* it. But (C) and (D) prove
that *in general* you need both: the spectral-shift case was lucky,
not representative. A correctness gate that ran only the residual
would pass a kernel off by a constant factor; one that ran only mass
conservation would pass a structurally wrong operator.

## (4) Recipe for a new operator-semigroup primitive

When adding a heat kernel / diffusion operator / geodesic flow:

1. Identify the **defining evolution equation** `∂_t S = 𝒢 S`. Write
   the residual `∂_t S − 𝒢 S` (FD in t; FD or autograd for `𝒢`).
   Tabulate it over the relevant `(t, scale, …)` grid; confirm it
   sits at the FD floor. Tabulating (not spot-checking) localizes the
   safe-parameter regime.
2. Identify a **conserved quantity** `Q` (probability mass, trace,
   row sum, energy, symplectic form, …). Evaluate `Q(S_t) − Q`;
   confirm machine precision.
3. **Run both.** They are complementary; one is not a substitute for
   the other. If the primitive has more than one conservation law,
   each adds coverage.
4. **Keep them reference-free.** The point is to validate without a
   known-correct answer. A closed-form spot check is a *third*,
   weaker check (it only constrains the points you happen to
   evaluate) — use it as corroboration, not the primary gate.

## (5) Relationship to the C1 narrative

The previous handoff folded C7 into C1's §3.1 ("the methodology that
caught the bug"). With this strengthening, C7 stands on its own: the
graph-kernel instance + the complementarity result are not part of
the C1 story. The paper can present C7 as a short standalone
methodology subsection (§3.3) that the C1 bug-catch motivates and
the graph-kernel instance generalizes.

## (6) Paper section

**§3.3 — "A reference-free correctness gate for operator-semigroup
primitives."** Lead with the abstract two checks, use C1 as the
motivating find, use the graph-kernel complementarity table (B/C/D)
as the evidence that both checks are necessary. Short — half a
column.

## (7) Status update

| Claim | Was | Now |
|---|---|---|
| C7 (dual-check validation methodology) | 🟡 (demonstrated once, with the C1 bug) | 🟢 (abstracted into a reference-free method; generalized to the graph heat kernel across the continuous/discrete divide; complementarity of the two checks proven via two corruptions each invisible to one check and caught by the other) |
