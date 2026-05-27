# Research claims catalog

The "our research" items in `holonomy_lib` v0.5.0, with explicit
status fields. This is the source document for the strengthening
pass: each row gets a deep dive in
`notes/strengthening/<short-name>.md`, and the consolidated paper
draws from the strengthened evidence.

Schema for each entry:
- **Claim**: precise mathematical / engineering statement
- **Current evidence**: what we already have
- **Strengthening needed**: what would close the gap to publication
- **Paper section**: which paper section it belongs in
- **Status**: 🔴 not strengthened, 🟡 partially, 🟢 paper-ready

---

## C1. Heat-kernel autograd implementation pitfall (spectral-shift factor)

**Claim** (revised): The literature recursion
`k^{n+2}(t, r) = -exp(-n·t) / (2π·sinh r) · ∂_r k^n(t, r)`
(Grigor'yan-Noguchi 1998 *Heat Kernel on Hyperbolic Space*, Bull.
LMS 30(6); reaffirmed Naganawa 2018 arXiv:1807.05708 *Heat kernel
recurrence on space forms*) is **correct as stated**. The bug we
caught was an **autograd implementation pitfall**: writing
```python
grad = torch.autograd.grad(k_n(t, r).sum(), r)[0]
return -grad / (2*math.pi*torch.sinh(r))   # WRONG — drops exp(-n·t)
```
because PyTorch's `torch.autograd.grad` computes `∂_r k^n` only,
and the `exp(-n·t)` spectral-shift factor — multiplicative in the
mathematical recursion — has to be applied explicitly outside the
autograd call. The factor is "invisible" to autograd because it's a
constant in r.

This is a generalizable lesson for porting nuanced analytical
formulas to PyTorch autograd code: multiplicative constants in r
that depend on n (or t) must be applied by hand; autograd only
gives you the r-derivative.

**Current evidence**:
- `src/holonomy_lib/hyperbolic/heat_kernel.py` — implementation
- `notes/validation/heat_kernel_findings.md` — derivation + bug
  history
- `notes/validation/heat_kernel_results.md` — PDE-residual at FD
  noise floor for n ∈ {1, 2, 3, 5, 7, 9}; probability mass ≈ 1
  to machine precision

**Strengthening needed**:
1. **Identify the published reference(s)** that state the recursion
   without the spectral-shift factor (we need to cite the source of
   the bug, not just claim "some references"). Candidates to check:
   Davies (1989) §5.7; Grigor'yan-Noguchi (1998); review papers on
   hyperbolic heat kernels in ML.
2. **Formal derivation writeup**: a clean 1-page derivation from the
   operator chain showing the spectral-shift factor emerges from
   `D ∘ exp(-(2m+1)·t) = exp(-(2m+1)·t) · D` (commutation of the
   shift operator with `(1/sinh r ∂_r)`).
3. **GPU benchmark** of the corrected recursion at multiple `(n, t, B)`
   sizes; demonstrate practical viability beyond the toy test cases.
4. **Reference-table comparison**: compare to tabulated values from
   numerical solvers (e.g. Crank–Nicolson on the radial heat eq.) at
   a few `(n, t, d)` triples for n=5, 7, 9.

**Paper section**: §3 "Implementation correctness"

**Status**: 🟢 **paper-ready** — see
`notes/strengthening/C1_C2_heat_kernel_strengthened.md`. Literature
attributed (Grigor'yan-Noguchi 1998); sympy-verified algebraic
identity (`notes/verification/heat_kernel_recursion_sympy.py` —
residual = 0 for corrected, residual ≠ 0 for naive); numerical
fingerprint locked in (naive value is exactly `exp(n·t)` off);
CPU performance characterized (GPU pending hardware).

---

## C2. Closed-form n=5 hyperbolic heat kernel

**Claim**: The H^5 heat kernel admits the hand-derived closed form
```
k^5_t(r) = (4πt)^(-5/2) · exp(-4t - r²/4t)
                  · [r²·sinh r + 2t·(r·cosh r − sinh r)] / sinh³ r
```
derived by applying `(1/sinh r · ∂_r)²` to `exp(-4t - r²/4t)`
analytically. Using this form directly (rather than two autograd-
based recursion steps from n=3) is faster AND ~3 orders of magnitude
more precise.

**Current evidence**:
- `src/holonomy_lib/hyperbolic/heat_kernel.py:_heat_kernel_unit_n5`
- 3 tests (closed-form vs recursion, r=0 limit, backward-finite)
- PDE-residual 1e-8..1e-6 (vs 1e-5..1e-3 for autograd path)

**Strengthening needed**:
1. **Derivation writeup**: step-by-step expansion of the operator
   chain. Already partially in the docstring; needs a clean note.
2. **Independent cross-check**: verify formula against an external
   reference (textbook, numerical solver). The H^5 closed form *is*
   in the literature (Grigor'yan's book Thm 8.21), but the specific
   polynomial expansion we use is hand-derived.
3. **GPU benchmark**: forward + backward latency at multiple `(t, r)`
   tensor sizes; compare to recursion path.
4. **Same approach for n=7?** Tractable polynomial expansion. The
   precision payoff would cascade.

**Paper section**: §4 "Precision/speed optimization"

**Status**: 🟢 **paper-ready** — see
`notes/strengthening/C1_C2_heat_kernel_strengthened.md`. Sympy
verifies (A) closed form ≡ corrected recursion, (B) closed form
satisfies H^5 heat equation, (C) r=0 limit matches docstring
formula (`notes/verification/heat_kernel_n5_sympy.py`, all
residuals = 0 algebraically). Performance characterized.

---

## C3. Autograd-safe arcsinh reparameterization for hyperbolic
distance/log

**Claim**: Replacing the textbook `d_k(x, y) = (1/√|k|) · arccosh(k·⟨x,y⟩_M)`
with the equivalent `d_k(x, y) = (2/√|k|) · arcsinh(√|k|·‖y - x‖_M/2)`
eliminates **two distinct backward-NaN failure modes** that the
textbook form has at x ≈ y:
1. `arccosh`'s derivative singularity at z = 1 (`1/√(z²-1) = ∞`).
2. Catastrophic cancellation in `‖u‖² = (k·⟨x,y⟩_M - 1)/k` when
   `k·⟨x,y⟩_M ≈ 1`.

The arcsinh form uses `‖y - x‖_M²` which is computed from
coordinate differences (no near-1 subtraction), and `arcsinh` is
entire (no derivative singularity). Combined with the
`_safe_sqrt(diff_sq)` autograd-safe helper, the result is
forward-AND-backward finite at every input.

**Current evidence**:
- `src/holonomy_lib/manifolds/lorentz.py:distance` and `:log`
- `notes/validation/autograd_safe_vs_geoopt.py` —
  comparison against eps-clamp pattern
- `tests/manifolds/test_lorentz.py::TestAutogradFinite` (~11 tests)

**Strengthening needed**:
1. **Formal proof** that the arcsinh form gives a finite gradient at
   all boundary inputs (currently shown empirically; needs symbolic
   verification via sympy).
2. **Numerical worked example**: show the catastrophic cancellation
   step-by-step. Compute float64 values at d = 1e-5 with both
   formulas, show the bit-level disagreement.
3. **Quantify the win in real training**: run a downstream task
   (Poincaré-embedding-style hierarchy embedding) with both
   variants, measure NaN-rate and final loss.
4. **Cite related work**: the `_safe_*` idiom is folklore in
   PyTorch (geoopt's eps-clamp; pytorch3d's similar patterns).
   Place our arcsinh-reparameterization in that lineage.

**Paper section**: §3 "Implementation correctness" — best-practices
sub-section

**Status**: 🟢 **paper-ready** — see
`notes/strengthening/C3_arcsinh_strengthened.md`. Sympy-verified
algebraic equivalence
(`notes/verification/arcsinh_reparam_sympy.py`); worked float64
example demonstrating textbook form loses all precision below
α ~ 1e-8, eps-clamp returns a constant 4.5e-3 below the clamp,
arcsinh tracks to fp precision
(`notes/strengthening/C3_arcsinh_worked_example_results.md`);
related-work positioning (geoopt / pytorch3d eps-clamp folklore);
performance characterized.

---

## C4. κ-can-cross-zero dynamic dispatch

**Claim**: For learnable κ on the κ-stereographic model, the
branch (spherical/hyperbolic/Euclidean) need NOT be locked at
construction. `torch.where(κ > 0, spherical_formula(√|κ|·α),
hyperbolic_formula(√|κ|·α))` with `|κ|` clamped at `finfo.tiny` for
the sqrt gives:
- Forward continuous through κ = 0 (both formulas reduce to the
  Euclidean limit at scaled-arg = 0).
- Backward finite at κ = 0 (the clamp keeps √|κ| ≠ 0; mask-by-cond
  gradients cleanly).

SGD can push κ through 0 without breakdown.

**Current evidence**:
- `src/holonomy_lib/manifolds/stereographic.py` —
  dynamic dispatch in `_tan_kappa_c` / `_atan_kappa_c`
- 2 tests (κ crosses 0 in either direction during SGD)

**Strengthening needed**:
1. **Stress test**: trajectory that crosses 0 multiple times. Does
   it stay stable?
2. **Comparison to alternatives**: (a) static-branch lock + restart;
   (b) Taylor-blended unified κ-trig. Quantify when each matters.
3. **GPU benchmark**: dispatch overhead. The `torch.where` evaluates
   both branches — does this double the compute on GPU?
4. **No clear prior art**: most ML libraries lock to one geometry
   per layer (κ-stereographic transformer heads). Our dynamic
   dispatch is a separate primitive.

**Paper section**: §4 "Precision/speed optimization" — sub-section
on learnable-geometry training

**Status**: 🟢 **paper-ready** — see
`notes/strengthening/C4_kappa_crossing_strengthened.md`. Sympy
verifies analytic continuation across κ = 0 (Taylor series match
term-by-term on both sides, limits/derivatives agree, integral
representation bridges the sign-conditional;
`notes/verification/kappa_crossing_sympy.py`). Numerical stress
test (`notes/strengthening/C4_kappa_crossing_stress.py` +
`_results.md`) demonstrates: 100-step SGD trajectory with 4
κ-sign crossings stays finite; static-branch lock has 1.38%
relative-error failure mode after a flip; Taylor-truncation
alternative is dominated by the dispatch across the entire
manifold domain; CPU latency overhead ~1.2–1.4× vs the
float-locked fast path; AMD ROCm GPU benchmark (Radeon RX 9060 XT)
confirms ×1.27 (float64) / ×1.58 (float32) overhead.

---

## C5. Per-point κ continuous (vs. GraphMoRE's discrete gating)

**Claim**: `HeterogeneousKappaManifold` exposes each point's κ as a
continuous real-valued tensor, vs. GraphMoRE's mixture-of-experts
gating over a discrete set of constant-curvature spaces. The
continuous parameterization:
1. Enables direct SGD on κ (autograd through every op).
2. Allows arbitrary κ values learned from data.
3. Loses the "interpretability of discrete gating" — there's no
   structural "this concept is in expert 3" output, just a
   continuous κ value per point.

**Current evidence**:
- `src/holonomy_lib/manifolds/heterogeneous_kappa.py`
- 26 tests (homogeneous-case agreement, heterogeneous behavior,
  autograd through both v + κ, combiner override)

**Strengthening needed**:
1. **Downstream-task ablation**: synthetic graph with known per-node
   curvature (e.g. mixture of trees + cycles, each region a
   different "true κ"). Train continuous per-point κ vs. discrete
   gating (a GraphMoRE-style implementation we'd need to add).
   Measure recovery quality + final loss.
2. **Pair-κ combiner study**: which combiner works best in practice?
   The arithmetic mean default is defensible but a real ablation
   would show whether harmonic mean or other choices help.
3. **Theoretical analysis**: when does a continuous κ field
   approximate a discrete mixture? Conditions under which they
   agree / disagree.

**Paper section**: §5 "New primitives" — the heterogeneous-κ
sub-section

**Status**: 🔴 implementation + tests exist; needs downstream-task
ablation + combiner study

---

## C6. Pair-κ combiner abstraction

**Claim**: The rule for combining two per-point κ's into an
effective pair-κ for pairwise distance is **not standardized in
the literature**. We propose a `Callable[[κ_x, κ_y], κ_eff]`
abstraction with two built-in defaults (arithmetic mean, harmonic
mean) and arbitrary callable override.

**Current evidence**:
- `src/holonomy_lib/manifolds/heterogeneous_kappa.py`:
  `_combiner_arithmetic_mean`, `_combiner_harmonic_mean`
- Tests verify homogeneous-case agreement (κ_x = κ_y reduces to
  `KappaStereographicManifold(κ).distance`) for both combiners

**Strengthening needed**:
1. **Theoretical motivation**: what properties should a "good"
   combiner have? (Commutative, smooth, recovers single-κ limit,
   behaves well at sign mix.) Justify the choices.
2. **Empirical comparison**: on the downstream task from C5, which
   combiner gives best recovery? Different combiners encode
   different inductive biases about "what the geometry looks like
   between heterogeneous regions."
3. **Mathematical relation to other approaches**: is the
   arithmetic-mean combiner equivalent to anything in the
   geometric-mean / wrapped-Cauchy / similar literature?

**Paper section**: §5 "New primitives" — same sub-section as C5

**Status**: 🔴 implementation + interface; needs theoretical /
empirical motivation

---

## C7. Validation methodology

**Claim**: For numerical implementations of manifold-PDE solutions
(heat kernels, geodesic flows, etc.), **two complementary
consistency checks** — (1) heat-equation PDE residual via finite
differences, (2) probability-mass normalization via Gauss-Legendre
quadrature — provide a robust correctness check that catches errors
the standard "compare to closed form at a few points" approach
misses. Demonstrated by finding the spectral-shift bug (C1).

**Current evidence**:
- `notes/validation/heat_kernel_validation.py` — the methodology
  applied
- The bug we found via this methodology

**Strengthening needed**:
1. **Apply the methodology to other primitives**: graph Laplacian
   heat kernel? Spherical heat kernel?
2. **General writeup**: methodology section that's reusable for
   any future manifold-PDE primitive.

**Paper section**: §3 "Implementation correctness" — methodology
sub-section

**Status**: 🟡 demonstrated once (with the bug catch); needs
generalization

---

## Strengthening pass — order of attack

By **paper-impact × tractability**:

1. **C1 + C7 first** (heat-kernel bug + methodology) — the
   strongest story, biggest paper anchor. ~half-day each.
2. **C2 next** (closed-form n=5) — natural extension of C1.
3. **C3 third** (arcsinh idiom) — broadly applicable, easy to
   strengthen with sympy + a numerical example.
4. **C4 fourth** (κ-sign crossing) — depends on having a downstream
   task to validate against.
5. **C5 + C6 last** (per-point κ + combiner) — requires a
   GraphMoRE-style baseline implementation for comparison; biggest
   work.

The paper draft (T#60) consumes all of the above. Order: build the
strengthening artifacts first, then the paper draws from them.
