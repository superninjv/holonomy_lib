# Changelog

All notable changes to `holonomy_lib` are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
version numbers follow [Semantic Versioning](https://semver.org).

## [0.2.1] - 2026-05-27

Packaging-metadata-only release. No code changes.

### Changed

- `pyproject.toml` description synced with the GitHub repo description:
  adds cellular sheaves, SO(3) Lie primitives, and information geometry
  to the listed capabilities (these modules shipped in 0.2.0 but the
  description hadn't caught up).

## [0.2.0] - 2026-05-27

Six new modules and several extensions on top of the v0.1.0 seed.
Tests: 543 → 613 passing. Module count: 6 → 12.

### Added: new modules

- **`optimization`**: Riemannian optimizers wrapping the existing
  manifold `projection` + `retraction` API.
  - `RiemannianSGD(manifold, lr)` + functional `riemannian_sgd_step`.
  - Works with `FixedRankManifold` (SVD-triple state) and
    `SPDManifold` (square-matrix state).
  - Adam / RMSProp deliberately omitted: adaptive preconditioning is
    user-side ergonomics, not part of the math of optimization on a
    manifold. Compose your own from `manifold.projection` +
    `retraction` + `torch.optim.Adam` buffers.
  - Refs: Absil-Mahony-Sepulchre (2008) §4.1; Bonnabel (2013).

- **`simplicial`**: Simplicial complex data structures + boundary
  operators + Vietoris-Rips construction.
  - `DenseSimplicialComplex`: batched, padded `(B, n_k_max, k+1)`
    simplex tables with validity masks. `boundary(k)` returns the
    dense Koszul-signed boundary matrix.
  - `SparseSimplicialComplex`: single-instance, sparse-CSC boundary
    matrices for the persistent-homology reduction kernel.
  - `vietoris_rips_dense` (batched) + `vietoris_rips_sparse`
    (single-instance) + `pairwise_distances`.
  - Refs: Munkres (1984); Hausmann (1995); Bauer (2021).

- **`topology`**: Hodge Laplacians + Betti numbers + batched
  persistent homology on simplicial complexes.
  - `hodge_laplacian(complex, k)`: `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`,
    kernel-dim = k-th Betti.
  - `betti_numbers(complex, max_dim)`: near-zero eigenvalue counting
    on each `L_k`. Closed-form verified: S¹ → (1, 1), S² → (1, 0, 1),
    T² → (1, 2, 1).
  - `persistence_diagrams(points, max_dim=2, max_radius=inf)`: batched
    H₀ + H₁ + H₂ on Vietoris-Rips filtrations. H₀ via union-find on
    sorted edges; H₁/H₂ via Z/2 left-to-right boundary-matrix reduction
    with Bauer-Kerber-Reininghaus clearing.
  - `reduction_backend="torch"` runs end-to-end on the filtration's
    device (CPU or GPU); current path is a same-algorithm torch port
    (not yet a custom CUDA kernel) and is ~21× slower than CPython
    sets at n=80; the GPU win is a v0.3 follow-up.
  - Refs: Eckmann (1944); Lim (2020); Schaub et al. (2020);
    Edelsbrunner-Letscher-Zomorodian (2002); Cohen-Steiner-Edelsbrunner-
    Harer (2007) stability.

- **`info_geometry`**: divergences on probability distributions.
  - `bregman_divergence(p, q, potential)` for any convex generator.
  - `kl_divergence_categorical(p, q)`, `kl_divergence_gaussian(...)`
    (Cholesky-stable closed-form).
  - Fisher information metric + natural gradient added in the v0.1
    roadmap sweep.
  - Refs: Bregman (1967); Banerjee et al. (2005); Amari (2016);
    Cover-Thomas (2006).

- **`sheaf`**: cellular sheaves on graphs and their Laplacians.
  - `GraphSheaf` dataclass; `sheaf_coboundary`, `sheaf_laplacian`
    (`δ^T δ`, PSD), `sheaf_dirichlet_energy`.
  - Reduces to the standard combinatorial graph Laplacian under
    trivial stalks. Orientation-flip on a 3-cycle correctly drops
    kernel dim 1 → 0 (the monodromy test).
  - v1 is dense-only with `SHEAF_DENSE_BYTES_CAP = 2 GiB` pre-flight
    guard; node-edge sheaves on graphs only (higher-dim cellular
    sheaves on simplicial complexes planned).
  - Rejects self-loops + duplicate edges at construction (callers
    must pre-process).
  - Refs: Hansen-Ghrist (2019); Bodnar et al. (2022) Neural Sheaf
    Diffusion; Curry (2014).

- **`lie`**: SO(3) primitives + real spherical harmonics.
  - `so3.axis_angle_to_matrix` (Rodrigues), `matrix_to_axis_angle`
    (dual-branch log: trace-based away from π, quaternion-based
    near π).
  - `so3.so3_exp` / `so3_log` on so(3) (3×3 skew matrices).
  - `so3.random_so3(batch_size, generator)`: Haar-uniform via the
    quaternion-from-3-uniforms construction (Shoemake 1992); chi-squared
    sanity test with p < 1e-6 bound in the suite.
  - `so3.compose(R1, R2)`: group product.
  - `real_spherical_harmonics(directions, l_max)`: closed-form Y_lm
    for `l_max ≤ 4`. Per-l block norm preserved under SO(3) rotation
    (full mixing via Wigner-D matrices is a v0.3 follow-up).
  - `SO3_LOG_NEAR_PI_RAD = 1e-7` is empirically calibrated for
    **float64**; do not change to e3nn / pytorch3d's `1e-2` (correct
    for float32) without re-running the empirical comparison in the
    `so3.py` docstring.
  - Refs: Hall (2015) §3.1; Shoemake (1992); Edmonds (1957);
    Cohen et al. (2018) Spherical CNNs.

### Added: extensions to existing modules

- **`spectral` additions**:
  - `magnetic.*`: magnetic Laplacian for directed graphs (Furutani
    2020), with sign-magnetic extension for signed-directed graphs
    (Fiorini 2023; He et al. 2023).
  - `heat_kernel_chebyshev`: Hammond-Vandergheynst-Gribonval (2011)
    Chebyshev-polynomial heat kernel.
  - `effective_resistance`, `commute_time` (Klein-Randić 1993).
  - `diffusion_map` (Coifman-Lafon 2006).
  - Sparse-COO/CSR/CSC paths for all four Laplacian variants
    (combinatorial, symmetric-normalized, random-walk, Kunegis signed);
    end-to-end with sparse `lanczos_eigsh`.

- **`algebra` additions**:
  - `lanczos_eigsh(A, k, which="LA"|"SA", sigma=σ, n_iter=…)`: Lanczos
    top-k eigensolver with full reorthogonalization (Paige 1972).
  - `"SA"` mode is shift-and-invert (Ericsson-Ruhe 1980): factor once
    outside the iteration, each step is a `lu_solve` against
    `A − σI`. Raises `RuntimeError("shift-invert breakdown")` if σ
    coincides with an eigenvalue. For graph Laplacians (which have 0
    in spectrum), use a small negative shift.
  - Sparse-input dispatch on the same API.

- **`discrete_geometry` additions**:
  - `forman_ricci_simple` + `forman_ricci_augmented`: combinatorial
    Forman-Ricci curvature (Sreejith et al. 2016; Samal et al. 2018).
  - Performance: pair tiling (`SINKHORN_TILE_DEFAULT = 256`) +
    sync-cadence-every-8-iters → 13× speedup on Ollivier-Ricci at
    n=64 (22.6s → 1.7s).

- **`provenance` additions**:
  - Class-method support for `FixedRankManifold` / `SPDManifold`
    methods; `self` is canonicalized into a provenance-signature
    dict. `replay()` raises `NotImplementedError` for class-method
    nodes (the canonicalization isn't reversible) and for
    tuple-of-tensors inputs (unpacked `point[i]` keys don't
    reassemble); both have explicit error messages.
  - `blake3` opt-in (faster hash), hooks API, SAELens-style dataset
    emission, diff API, persistence (`save` / `load`).

### Changed

- **Self-loops dropped at entry to ALL graph primitives** (codified
  in `CONVENTIONS.md`). Sheaf v1 also rejects duplicate edges at
  `GraphSheaf` construction time; call sites must pre-process.
- `FixedRankManifold.retraction` auto-switches to Halko-Martinsson-
  Tropp randomized SVD at low `r/min(m, n)` ratios (the common case);
  25× speedup at 1024×1024×32 (193 ms → 7.6 ms).

### Notes

- The `v0.1.0` git tag was placed mid-development, before the full
  "v0.1 roadmap" cycle completed. Everything from `optimization`
  onward in the list above actually ships in this `v0.2.0` release.

## [0.1.0] - 2026-05-26

Initial public release. Six seed modules:

- **`manifolds`**: `FixedRankManifold` (Vandereycken 2013),
  `SPDManifold` (Pennec et al. 2006 affine-invariant metric).
- **`algebra`**: `truncated_svd` exact (Eckart-Young) + randomized
  (Halko-Martinsson-Tropp 2011).
- **`tensor_calculus`**: `hosvd`, `mode_product`, `mode_unfolding`
  (Kolda-Bader 2009).
- **`spectral`**: combinatorial / symmetric-normalized / random-walk
  / signed Laplacians (Chung; von Luxburg; Kunegis); batched
  `laplacian_eigenmaps` embedding.
- **`discrete_geometry`**: `ollivier_ricci_curvature` via batched
  log-domain Sinkhorn on all-pairs shortest paths (Ollivier 2009;
  Cuturi 2013); `discrete_ricci_flow` + `ricci_flow_with_surgery`
  ("Perelman on networks", Sia et al. 2019; Ni-Lin-Luo-Gao 2019).
- **`provenance`**: content-addressable Merkle DAG of math
  primitives, `@with_provenance` decorator, `record()` context
  manager, `ProvenanceRegistry` with substitution / replay /
  SAELens-style emission for mechanistic interpretability.

- `audit.py`: build-gate enforcing the no-magic-numbers discipline:
  every numerical literal must be derived, a universal invariant, or
  cataloged in `notes/magic_numbers.md` with scale-of-validity.

[0.2.1]: https://github.com/superninjv/holonomy_lib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/superninjv/holonomy_lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/superninjv/holonomy_lib/releases/tag/v0.1.0
