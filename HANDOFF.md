# Handoff — holonomy-lib — initial build

You're the first session on this project. The repo was seeded by a session running on `synoros-substrate` (sibling repo at `~/projects/synoros-substrate/`). Read `README.md` first for project vision.

## Why this project exists

`synoros-substrate` keeps hitting the same problem: it needs mathematical primitives (Riemannian manifolds, persistent homology, spectral embeddings, information-geometric divergences, etc.) that aren't available in a single library. Each session has to either:

- Pull from one library (e.g., pymanopt) that's CPU-only and bottlenecks scaling
- Pull from several libraries with inconsistent conventions
- Implement primitives from scratch (slow, error-prone)
- Tolerate arbitrary numerical constants because the right derivations are deep in some paper

The result: drift. Sessions add `lr = 0.01` or `epsilon = 1e-5` because the principled value would require a research detour. The substrate accumulates technical debt that's hard to back out.

`holonomy-lib` is the library that, if it existed, would have prevented all of this.

## What to build

A PyTorch-native, GPU-capable, batched-first, audit-compliant math library covering the fields below. Each section names the existing libraries that cover slices of it (port from them where licenses allow, otherwise reimplement) and the synoros-substrate prior art that should migrate here.

### 1. Topology

**Scope**: simplicial complexes, persistent homology, bottleneck/Wasserstein distance between persistence diagrams, persistent Betti numbers, persistence images, topology-of-trajectories tools.

**Existing libraries to consolidate from**:
- `gudhi` — comprehensive C++/Python, slow for batched/GPU usage
- `ripser` — fast Vietoris-Rips, no GPU
- `giotto-tda` — sklearn-compatible, mostly CPU
- `torchph` — emerging PyTorch persistent homology (small community)

**synoros-substrate prior art**:
- `src/from_vtm/substrate/homology.py` — ripser-based, currently diagnostic-only

**What to do**:
- Build batched persistence-diagram computation on GPU when possible
- Persistence-image transform (Adams et al. 2017) as a PyTorch operator
- Bottleneck distance with autograd support (subdifferentiable)
- Cite: Edelsbrunner & Harer (2010), Adams et al. (2017), Cohen-Steiner et al. (2007)

### 2. Differential geometry

**Scope**: Riemannian manifolds (Stiefel, Grassmann, sphere, fixed-rank matrices, SPD, hyperbolic, Lorentzian, product manifolds), tangent spaces, projection/retraction, parallel transport, exponential/logarithmic maps, geodesics, curvature.

**Existing libraries to consolidate from**:
- `geoopt` — PyTorch-native Riemannian optimization; has Stiefel, Sphere, Hyperbolic, SPD, Product; MISSING fixed-rank matrices
- `geomstats` — broader, multi-backend (numpy/PyTorch/jax); has fixed-rank, more manifolds; CPU-heavier
- `pymanopt` — comprehensive but numpy-only

**synoros-substrate prior art**:
- `src/holonomy_library/manifolds/fixed_rank.py` — PyTorch-native FixedRankManifold with batched operations (Vandereycken 2013). **PORT THIS DIRECTLY.**

**What to do**:
- Move synoros-substrate's FixedRankManifold here as the first manifold
- Wrap geoopt's manifolds with consistent API + audit-compliant constants
- Implement what's missing: variable-rank manifolds (union of fixed-rank), stratified manifolds for rank changes, manifold-valued ODE integrators
- Cite: Vandereycken (2013), Absil-Mahony-Sepulchre (2008), Edelman-Arias-Smith (1998)

### 3. Algebra

**Scope**: linear algebra primitives (efficient, audit-clean), tensor algebra (Einstein notation helpers, contractions, mode products), group representations (cyclic, dihedral, permutation), Lie algebras (su(n), so(n), basic), exterior algebra (wedge products, forms).

**Existing libraries**:
- `torch.linalg` — covers most linear algebra; we wrap with audit-clean defaults
- `opt_einsum` — efficient einsum path optimization
- `sympy.combinatorics` — group theory (CPU-only)
- `lie-learn` — finite groups for ML (Cohen 2014)

**synoros-substrate prior art**:
- Various ad-hoc tensor manipulations in `src/synoros_substrate/v3/`

**What to do**:
- Audit-clean wrappers around `torch.linalg` (defaults documented, no hidden tolerances)
- Tensor decompositions: CP (Kolda 2009), Tucker, Tensor-Train, HOSVD — all batched on GPU
- Cite: Kolda & Bader (2009) for tensor decompositions, Cohen (2014) for group reps

### 4. Spectral theory

**Scope**: graph Laplacians (combinatorial, normalized, signed), eigendecomposition (full + truncated via Lanczos/Arnoldi), spectral embeddings, heat kernels, diffusion maps, effective resistance, commute time, signed graph spectra.

**Existing libraries**:
- `scipy.sparse.linalg` — sparse eigensolvers; CPU-only
- `torch.linalg.eigh` / `svd` — GPU for dense
- `cugraph` — NVIDIA-only GPU graph algorithms

**synoros-substrate prior art**:
- `src/synoros_substrate/v3/priors.py:spectral_prior_init` — per-relation Laplacian eigendecomposition
- effective-resistance distance computations in older revision2/iteration3 code

**What to do**:
- Per-relation / multi-layer Laplacian utilities, batched
- Lanczos solver in PyTorch for top-K eigenvectors of sparse Laplacians (GPU)
- Heat kernel on graph: `exp(-tL)` via expm or Lanczos
- Cite: Chung (1997) for spectral graph theory, Belkin & Niyogi (2003) for Laplacian eigenmaps

### 5. Information geometry

**Scope**: Fisher information metric, KL divergence (forward/reverse/symmetric), Bregman divergences, natural gradient, exponential-family geometry, mixture-model geodesics.

**Existing libraries**:
- `torch.distributions` — KL between built-in distributions
- `geomstats.information_geometry` — has some of this
- `pyro.ops` — Fisher information utilities

**synoros-substrate prior art**:
- FEP loss computation in `src/synoros_substrate/v3/fep_learner.py` — surprise + complexity (KL-like)
- β derivation in `src/synoros_substrate/v3/fep_learner.py:derived_beta`

**What to do**:
- Natural-gradient utilities (Fisher block-diagonal approximations, K-FAC-like)
- Bregman divergences for exponential families
- Manifold-valued KL (e.g., between distributions parameterized by points on Stiefel)
- Cite: Amari (1985), Amari-Nagaoka (2000) for information geometry; Friston (2010) for FEP

### 6. Tensor calculus

**Scope**: Einstein notation helpers, batched contractions, tensor decompositions, covariant/contravariant index management, Christoffel symbols, covariant derivatives, parallel transport on tensor fields.

**Existing libraries**:
- `opt_einsum` — Einstein contraction optimization
- `torch.einsum` — runtime
- `tensorly` — tensor decompositions (CP, Tucker, TT, etc.); has PyTorch backend but not GPU-first

**synoros-substrate prior art**:
- `src/synoros_substrate/v3/experiments/interaction_primitives/operators.py` — five interaction operators tested as bake-off (scalar dot, hadamard, outer, signed hadamard, cross-relation outer)

**What to do**:
- Tensor decomposition API: factor any rank-N tensor via CP/Tucker/HOSVD batched on GPU
- Manifold-valued tensor fields (e.g., tensors on a Riemannian manifold)
- Cite: Kolda & Bader (2009), Sidiropoulos et al. (2017) for tensor decompositions

### 7. Optimization

**Scope**: Riemannian SGD/Adam, manifold-aware gradient descent, constrained optimization on manifolds, trust-region methods, second-order methods (Newton on manifolds), bilevel optimization (for hyperparameter learning).

**Existing libraries**:
- `geoopt.optim` — Riemannian Adam, SGD on built-in manifolds
- `pymanopt.optimizers` — comprehensive but CPU
- `higher` / `torchopt` — bilevel / implicit-gradient methods

**synoros-substrate prior art**:
- `scripts/v3/auto_lambda_*.py` — three empirical-Bayes hyperparameter optimization attempts (bilevel, implicit grad, variational EM)

**What to do**:
- Riemannian optimizers that work on our fixed-rank manifold (and others)
- Trust-region methods on manifolds with proper step-size derivation
- Cite: Absil-Mahony-Sepulchre (2008) for Riemannian optimization

### 8. Probability

**Scope**: conjugate priors with closed-form posterior updates, manifold-valued distributions, empirical-Bayes hyperparameter inference, posterior sampling on manifolds, variational inference utilities.

**Existing libraries**:
- `torch.distributions` — flat-space distributions
- `pyro` — full PPL
- `geomstats` — some manifold-valued distributions

**What to do**:
- Conjugate-prior calculator (Beta-Bernoulli, Dirichlet-Multinomial, etc.) with audit-compliant defaults
- Bingham, von-Mises-Fisher distributions on spheres/Stiefel
- Empirical-Bayes utilities (MacKay's α update, proper variational forms)
- Cite: Murphy (2012, ch. 5) for conjugate priors; Mardia & Jupp (2000) for directional statistics

## Audit infrastructure

**Already ported**: `src/holonomy_lib/audit.py` (from synoros-substrate's `src/holonomy_library/audit.py`). Scans Python source for numeric literals; flags any that aren't:
- In ALLOWED_LITERALS (mathematical identities, universal unit conversions)
- Documented in `notes/magic_numbers.md` catalog

**Discipline for this library**: every numerical constant in library code must be:
1. **Derived from inputs** — function of dimensions, structure, problem parameters
2. **Universal invariant** — π, e, log N, 1/N, √N, mathematical identities
3. **Experimentally set with documented procedure** — value tuned via documented sweep, scale-of-validity recorded

The audit tool enforces this. CI integration is the first non-trivial task (run `holonomy_lib.audit` in pre-commit / GH Actions).

## Architectural principles

1. **PyTorch-native + GPU-capable** — every operation works on `torch.Tensor` on `cuda` device.
2. **Batched-first** — operations take leading batch dim; scalar/single ops are special cases.
3. **Type-annotated** — full type hints, including tensor shapes in docstrings.
4. **Cited** — every non-trivial operation references the paper/textbook for its math.
5. **Audit-compliant** — passes the audit; CI enforces.
6. **No "library defaults" with magic numbers** — if a function takes a parameter, the documentation derives the right value or marks it experimental.
7. **Composition over inheritance** — manifolds compose via products; optimizers compose via decoration; etc.

## Package structure (skeleton already in place)

```
holonomy-lib/
  README.md                   — vision + relationship to synoros-substrate
  HANDOFF.md                  — this file
  CLAUDE.md                   — operating constraints (for agents working here)
  pyproject.toml              — Python package config
  src/holonomy_lib/
    audit.py                  — discipline enforcement (ported)
    __init__.py
    manifolds/                — TO BUILD
      __init__.py
      fixed_rank.py           — PORT from synoros-substrate
      (more manifolds)
    topology/                 — TO BUILD
    algebra/                  — TO BUILD
    spectral/                 — TO BUILD
    info_geometry/            — TO BUILD
    tensor_calculus/          — TO BUILD
    optimization/             — TO BUILD
    probability/              — TO BUILD
  tests/                      — TO BUILD (mirror src/ structure)
  notes/
    magic_numbers.md          — catalog (ported, currently empty for this lib)
```

## Recommended first moves

In order:

1. **Set up the package** — finalize `pyproject.toml`, install in dev mode, verify imports work.
2. **Wire CI for the audit** — pre-commit hook or GH Action that runs `python -m holonomy_lib.audit src/` on every push. Failing audit = failing build.
3. **Port the first manifold** — copy `src/holonomy_library/manifolds/fixed_rank.py` from synoros-substrate, adapt namespace to `holonomy_lib.manifolds.fixed_rank`, write tests.
4. **Choose the second section to build out** — probably differential geometry (build out from the fixed-rank seed) OR spectral theory (we have prior art there too via the spectral_prior_init).
5. **Pick a citation/docs convention** — e.g., every public function has a `References:` section in docstring with paper title + year + author. Decide on the convention and apply consistently.

## Operating constraints (binding for agents working here)

1. **Discuss-look-test research method.** Before adding a primitive, find the existing implementations (geoopt, geomstats, etc.) and decide: port directly, wrap, or reimplement. Don't reinvent without checking.
2. **Every numerical constant has a derivation or catalog entry.** No exceptions.
3. **GPU-first design.** Default to batched-on-GPU; CPU and single-element are special cases. Verify shapes work for `B = 0`, `B = 1`, `B > 1`.
4. **Citations are non-optional.** Every operation references the math source. No "trust me" implementations.
5. **Tests before commit.** Unit tests for correctness, property tests where applicable (e.g., manifold operations preserve their invariants), comparison tests against established libraries where possible.
6. **No fluent prose without source-line verification.** Same constraint as synoros-substrate — if you find yourself writing confident prose about how a function should work, stop and read the file at the relevant lines.
7. **Library/architecture separation.** This IS the library. Don't put research-experiment code here; don't put architectural commitments here. This is purely tools.

## Relationship management

- `synoros-substrate` is the primary consumer. When something works here, synoros-substrate should be updated to depend on it and stop carrying its own implementation.
- Currently: `synoros-substrate/src/holonomy_library/` contains audit + some manifold code that should EVENTUALLY migrate here. When it does, synoros-substrate gets `holonomy-lib` as a dependency.
- For now: independent development. Migration happens as primitives stabilize.

## What to ignore

- Don't try to be a research framework (no training loops, no model classes, no architecture). Math primitives only.
- Don't try to be sklearn-compatible (different audience).
- Don't try to support TF or JAX (PyTorch only; let other libs do other backends).
- Don't add features speculatively. Only build what synoros-substrate or another consumer actually needs, motivated by a real research question.

## What success looks like

A year from now: a researcher building a new cognitive-substrate or geometric-ML project can `pip install holonomy-lib` and immediately have access to:
- All the manifolds they'd need without arbitrary defaults
- Tensor decompositions they don't have to re-derive
- Persistent-homology utilities that work on GPU
- Audit discipline that prevents the kind of drift that motivated this library
- Cited implementations they can trust as research-grade

Every operation in the library has a paper reference. Every numerical constant either derives from inputs or is documented as a posited choice with scale-of-validity. CI enforces this.

Good luck.

— synoros-substrate session, 2026-05-25
