# Open research / validation questions — hyperbolic extension

This catalogs implementation choices in the Stage 1–3 hyperbolic
work that are **not directly cited to a published formula**. They
work in our tests and against geoopt cross-comparisons, but they're
derivations we made or extensions of known results, so before
relying on them for novel research we should validate further.

Severity legend:
- **⚠️ research** — the math we wrote may not match the literature; needs
  cross-check against a published reference.
- **🔬 heuristic** — works but arbitrary; a more principled alternative
  exists in the literature.
- **📐 derivation** — math we derived ourselves from textbook identities;
  consistent with established forms but not directly cited.

---

## ⚠️ research-level items (revisit before publication)

### ~~Heat-kernel autograd recursion for n ≥ 5~~ (RESOLVED — major bug found and fixed)

- **What we thought**: `k^{n+2}(t, d) = -(2π sinh d)^{-1} · ∂_d k^n(t, d)`
  was the correct Grigor'yan-Noguchi recursion.
- **What we found**: That recursion is **wrong**. The correct form
  is `k^{n+2} = -exp(-n·t) · (2π sinh d)^{-1} · ∂_d k^n`, with the
  `exp(-n·t)` factor accounting for the spectral-bottom shift
  `((n+1)/2)² − ((n-1)/2)² = n` between dimensions.
- **How we found it**: `notes/validation/heat_kernel_validation.py`
  checks `∂_t k − Δ_radial k ≈ 0` and `∫ k_t · dV = 1`. The original
  recursion produced O(1) residuals for odd n ≥ 5 (vs FD noise floor
  for n = 1, 2, 3) — caught immediately.
- **Status now**: fixed. n ∈ {1, 2, 3, 5, 7, 9} all pass both
  validation checks. **Even n ≥ 4 now raises `NotImplementedError`**
  (the previous integral-then-recurse code path was also incorrect;
  the simple recursion doesn't compose with the Davies-Mandouvalos
  integral). Open follow-up: derive the correct even-n recursion.
- **Findings document**: `notes/validation/heat_kernel_findings.md`.
- **Paper potential**: now stronger. A general-n differentiable
  heat-kernel implementation is a useful primitive for downstream
  hyperbolic ML methods; we found a published-recursion error in the
  process. Workshop-paper-worthy.
- **Update (2026-05-28)**: even n ≥ 4 now also works via the
  corrected spectral-shift recursion seeded from the n=2 integral
  form. Probability mass = 1 to machine precision; heat-equation
  residual at quadrature-error scale (1e-3..1e-5). Closed-form n=5
  added, replacing the n=5 autograd-recursion path with a precise
  polynomial expression — ~3 orders of magnitude tighter PDE
  residual (1e-8..1e-6 vs 1e-5..1e-3) and faster (no autograd
  calls). The n=7 recursion now seeds from the closed-form n=5,
  cascading the precision win.

### Frechet mean on `KappaStereographicManifold` for κ > 0 (spherical)

- **Where**: `src/holonomy_lib/hyperbolic/frechet_mean.py`
- **What**: Karcher iteration is *guaranteed* to converge on Hadamard
  manifolds (negative sectional curvature — our hyperbolic case). For
  κ > 0 (spherical), convergence holds only when points lie in a
  "small enough" neighborhood (within the injectivity radius).
- **Risk**: for spherical inputs that span more than a hemisphere,
  the iteration may not converge or may converge to a non-unique
  local optimum.
- **Validation needed**:
  - Add explicit test on the spherical branch with wide point spread.
  - Document the safe input regime (injectivity radius is `π/√κ`).
- **Mitigation**: undocumented currently. **TODO**: add a docstring
  caveat.

### `KappaStereographicManifold.random_point` clip at π/(8√κ)

- **Where**: `src/holonomy_lib/manifolds/stereographic.py:random_point`
- **What**: Spherical branch caps `‖v‖` at `π/(8·√κ)` before `exp_0`.
- **Status**: heuristic — works for sampling test points well-inside
  the domain, but isn't a published sampling scheme.
- **Better alternatives** (planned):
  - **Wrapped normal on the manifold**: `exp_0(N(0, σ²·I))` with `σ²`
    chosen as a hyperparameter, then reject outliers. Standard in
    e.g. Skopek et al. 2019 (Mixed-curvature VAE).
  - **Uniform on the geodesic ball**: sample direction uniform on
    `S^{n-1}`, radius uniform on `[0, r_max]`.
- **Risk**: low — the function is only used for testing, not for any
  statistically-meaningful sampling task downstream.

### `manifold_aware_inner` design choice

- **Where**: `src/holonomy_lib/hyperbolic/manifold_inner.py`
- **What**: We define `manifold_aware_inner(x, y) = ⟨log_o(x),
  log_o(y)⟩_o` (Riemannian inner of log-at-origin tangents).
  Satisfies `inner(x, x) = d(o, x)²`.
- **Alternatives in the literature**:
  - **Parallel transport-based**: `⟨v, PT_{x→y}(w)⟩_y` for some
    selected `v, w` — more principled for "similarity along a
    geodesic" but caller must pick the base.
  - **Klein-projective inner** (hyperbolic specific): the
    cross-ratio-based form that's preserved under Möbius
    transformations. Different invariance properties.
  - **Simple cosine**: `cos∠(log_o(x), log_o(y)) = inner /
    (‖log_o(x)‖ · ‖log_o(y)‖)` — what users probably want for
    "similarity" tasks.
- **Status**: our choice is one of several valid options. The
  literature doesn't have a single canonical "manifold-aware inner
  product."

---

## 📐 derivations from textbook identities (consistent but un-cited)

### Lorentz `log` via `arcsinh(arg)/(arg·√(arg²+1))·u` form

- **Where**: `src/holonomy_lib/manifolds/lorentz.py:log`
- **What**: We reparameterize `log_x(y) = (α/sinh α)·u` via
  `α = 2·arcsinh(arg)`, `arg = √|k|·‖y-x‖_M/2`, giving
  `α/sinh(α) = arcsinh(arg)/(arg·√(arg²+1))`.
- **Status**: derivation from textbook identities (Cannon et al. 1997
  §3 half-angle, Lee 2018 §5). Not cited as a single formula but
  algebraically equivalent to the textbook `β·u/‖u‖` form.
- **Verification**: passes `log_exp_inverse` tests at atol=1e-10;
  cross-checked against geoopt at unit |k| to atol=1e-10.

### Lorentz `distance` via `2·arcsinh(√|k|·‖y-x‖_M/2)`

- **Where**: `src/holonomy_lib/manifolds/lorentz.py:distance`
- **What**: Half-angle identity `cosh(α) - 1 = 2·sinh²(α/2)` applied
  to the textbook `(1/√|k|)·arccosh(k·⟨x,y⟩_M)`.
- **Status**: standard half-angle identity, specific application choice
  for autograd-stability at `x ≈ y`. Verified to atol=1e-6 against
  geoopt at non-unit |k|, atol=1e-10 at unit |k|.

### Lorentz exp/log/distance at general k ≠ -1

- **Where**: `src/holonomy_lib/manifolds/lorentz.py`
- **What**: Generalized the unit-curvature formulas to general `k < 0`
  via the scaling identity (sectional curvature ↔ hyperboloid radius).
- **Status**: standard scaling; not all textbooks make the
  parameterized form explicit, but geoopt does it correctly and
  matches our results.

### Heat kernel curvature scaling

- **Where**: `src/holonomy_lib/hyperbolic/heat_kernel.py:hyperbolic_heat_kernel`
- **What**: `k^n_{−K, t}(d) = K^{n/2} · k^n_{−1, K·t}(√K·d)`.
- **Status**: derived from the heat-equation invariance under
  rescaling + the probability-density normalization. Standard scaling
  argument; not always cited directly.

### `_safe_sqrt` / `_safe_sinhc` / `_safe_arcsinhc` autograd-safe idiom

- **Where**: `src/holonomy_lib/manifolds/lorentz.py:60-130`
- **What**: `torch.where(cond, formula_at_safe, default)` pattern
  where the formula branch uses the where-substituted input (never the
  singular value).
- **Status**: standard PyTorch autograd-safety idiom (geoopt, pytorch3d,
  and PyTorch's own docs use the same approach). Our specific helpers
  for sqrt, sinhc, arcsinhc, tanhc, tanc, atanhc, atanc are novel
  combinations but the idiom is well-established.

---

## 🔬 heuristic choices (works, but other choices are equally valid)

### `_safe_atanhc` substitution value of 0.5

- **Where**: `src/holonomy_lib/manifolds/stereographic.py:_safe_atanhc`
- **What**: When `t ≤ 0`, we substitute `t_safe = 0.5` (instead of `1`
  used elsewhere).
- **Reason**: `arctanh(t)` blows up at `t = 1`, so we use a value
  strictly inside the domain. Any value in `(0, 1)` works.
- **Risk**: none — the where-mask ensures the substituted value never
  affects forward output. Low priority.

### `hyperbolic_laplacian_eigenmaps` fail-loud error suggestion

- **Where**: `src/holonomy_lib/hyperbolic/laplacian_eigenmaps.py`
- **What**: On divergence we raise `RuntimeError("...retry with a
  smaller lr (an order of magnitude smaller is usually enough)...")`.
- **Status**: heuristic suggestion. A more principled approach would be
  adaptive lr-shrinking on detection (Armijo backtracking style), but
  that adds complexity for what's typically a misconfiguration error.

---

## Items NOT requiring further research

These are well-established and don't need extra validation:

- `LorentzManifold` core API (Nickel-Kiela 2018; Chen et al. 2022 HyboNet;
  Cannon et al. 1997). Unit-curvature formulas are canonical.
- `KappaStereographicManifold` core API at non-zero κ (Bachmann et al.
  2020). Möbius gyro-algebra is Ungar 2008.
- `RiemannianSGD` integration (Absil-Mahony-Sepulchre 2008 §4.1).
- Karcher iteration for Fréchet mean on Hadamard manifolds (Karcher
  1977; Afsari 2011).
- Davies-Mandouvalos closed form for `H^3` heat kernel (1988).
- Random-walk normalization of the graph Laplacian (von Luxburg 2007).
- Gauss-Legendre quadrature with `s = d + u²` change of variable for
  the n=2 Davies-Mandouvalos integral (Atkinson 1989 §5.6).
